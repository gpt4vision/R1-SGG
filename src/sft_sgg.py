
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

"""
import json
import random
from tqdm import tqdm
import torch
import math

from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from qwen_vl_utils import process_vision_info

PROMPT2 = """Generate a structured scene graph for an image of size ({width} x {height}) using the following format:

```json
{{
  "objects": [
    {{"id": "object_name.number", "bbox": [x1, y1, x2, y2]}},
    ...
  ],
  "relationships": [
    "[subject] -> [relation type] -> [object]"
    ...
  ]
}}
```

### **Guidelines:**
- **Objects:**
  - Assign a unique ID for each object using the format `"object_name.number"` (e.g., `"person.1"`, `"bike.2"`).
  - Provide its bounding box `[x1, y1, x2, y2]` in integer pixel format.
  - Include all visible objects, even if they have no relationships.

- **Relationships:**
  - Each relationship item should be a triplet: [subject] -> [predicate] -> [object].
  - Use active voice (e.g., "person.1 -> riding -> bike.2" instead of "bike.2 -> ridden by -> person.1").
  - Omit relationships for orphan objects.

### **Example Output:**
```json
{{
  "objects": [
    {{"id": "person.1", "bbox": [120, 200, 350, 700]}},
    {{"id": "bike.2", "bbox": [100, 600, 400, 800]}},
    {{"id": "helmet.3", "bbox": [150, 150, 280, 240]}},
    {{"id": "tree.4", "bbox": [500, 100, 750, 700]}}
  ],
  "relationships": [
    "person.1 -> riding -> bike.2",
    "person.1 -> wearing -> helmet.3",
  ]
}}
```

Now, generate the complete scene graph for the provided image:
"""


def format_answer(objects:str, relationships:str, shuffle=False):
    objects = json.loads(objects) # a list of {"id": xxx, "bbox": xxx}
    relationships = json.loads(relationships)
    if shuffle:
        random.shuffle(objects)

        obj_map = {}
        new_objects = []
        for new_idx, obj in enumerate(objects):
            name, old_idx = obj["id"].split('.')
            bbox = obj["bbox"]

            new_obj = '%s.%s'%(name, new_idx+1)
            obj_map[obj["id"]]  = new_obj

            new_objects.append({"id": new_obj, "bbox": bbox})

        new_rels = []
        for r in relationships:
            sub = obj_map[r["subject"]]
            obj = obj_map[r["object"]]
            rel = r["predicate"]
            tmp = {"subject": sub, 
                   "predicate": rel,
                   "object": obj 
                   }

            new_rels.append(tmp)
        objects, relationships = new_objects, new_rels


    objects = [json.dumps(e) for e in objects]
    relationships = [json.dumps(e) for e in relationships]

    # Format structured answer
    structured_answer = (
        "```json\n"
        "{\n"
        "  \"objects\": [\n" + ",\n".join(objects) + "\n  ],\n"
        "  \"relationships\": [\n" + ",\n".join(relationships) + "\n  ]\n"
        "}\n"
        "```\n"
    )
    return structured_answer


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def format_data(sample, shuffle=False):
    """Prepare dataset example for training."""

    image = sample["image"].convert('RGB')
    iw, ih = image.size
    prompt = sample['prompt_open']
    #prompt = PROMPT2.format(width=iw, height=ih)    

    #normalize box to [0, 1000]
    objs = []
    for obj in sample['objects']:
        box = obj['bbox']
        obj['bbox'] = [int(box[0]/iw*1000), int(box[1]/ih*1000),
                       int(box[2[/iw*1000), int(box[3]/ih*1000)]
        objs.append(obj)

    answer = format_answer(objs, sample["relationships"], shuffle=shuffle)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful and multimodal AI assistant."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]
    return {"messages": messages}


def main():
    accelerator = Accelerator()
    # args
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # load dataset 
    train_dataset = load_dataset(script_args.dataset_name)['train']
    #split_db = dataset.train_test_split(test_size=0.01, seed=42)
    #train_dataset = split_db["train"]
    #val_dataset = split_db["test"]
    print(f"Training set size: {len(train_dataset)}")
    #print(f"Validation set size: {len(val_dataset)}")
    print("Train set[0]:", format_data(train_dataset[0]))

    
    # model config.
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    min_pixels = 3136
    max_pixels = 1024 * 28 * 28
    processor = Qwen2VLProcessor.from_pretrained(model_args.model_name_or_path, 
                    min_pixels=min_pixels, max_pixels=max_pixels)



    class Collator(object):
        def __init__(self, processor):
            self.processor = processor
            self._db = {}

        def __call__(self, examples):
            # Get the texts and images, and apply the chat template
            texts, image_inputs = [], []
            for example in examples:
                if str(example) not in self._db:
                    self._db[str(example)] = 0

                shuffle = (self._db[str(example)] > 0) & (random.random() > 0.5)
                format_example = format_data(example, shuffle)['messages']
                self._db[str(example)] += 1

                text = self.processor.apply_chat_template(format_example, tokenize=False)
                image_input = process_vision_info(format_example)[0]
                texts.append(text)
                image_inputs.append(image_input)
    
            # Tokenize the texts and process the images
            batch = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
            # The labels are the input_ids, and we mask the padding tokens in the loss computation
            labels = batch["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100  #
            # Ignore the image token index in the loss computation (model specific)
            if isinstance(self.processor, Qwen2VLProcessor):
                image_tokens = [151652,151653,151655]
            else:
                image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)]
            for image_token_id in image_tokens:
                labels[labels == image_token_id] = -100
            batch["labels"] = labels
    
            return batch

    ################
    # Training
    ################
    try:
        rank = torch.distributed.get_rank()  # GPU ID or node rank
        world_size = torch.distributed.get_world_size()  # Total number of GPUs/nodes

        global_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * world_size
        )
        total_steps = len(train_dataset) // global_batch_size * training_args.num_train_epochs
        print("*"*100, "\nglobal_batch_size:", global_batch_size, " total steps:", total_steps, "\n", "*"*100)
    except:
        pass

    training_args.gradient_checkpointing_kwargs={"use_reentrant": False}
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.dataset_text_field=""

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=None, #val_dataset,
        processing_class=processor.tokenizer,
        data_collator=Collator(processor),
        peft_config=get_peft_config(model_args),
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    main()
