
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
import os
import json
import random
from tqdm import tqdm
import torch
import math
from dataclasses import dataclass, field
import re
import glob

from accelerate import Accelerator
from datasets import load_dataset, load_from_disk

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

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

#---------------------- prompt templates ----------------------------
from open_r1.trainer.utils.prompt_gallery import PROMPT_SG, PROMPT_CLOSE, PROMPT_CLOSE_PSG, PROMPT_CLOSE_VG150 
#---------------------------------------------------------------------------


def format_answer(objects:str, relationships:str, shuffle=False):
    if isinstance(objects, str):
        objects = json.loads(objects) # a list of {"id": xxx, "bbox": xxx}
    if isinstance(relationships, str):
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


def replace_answer_format(item: str) -> str:
    return item.replace("<answer>", "```json").replace("</answer>", "```")

def format_data(dataset_name, sample, use_predefined_cats=False, remove_image_size_in_prompt=True, shuffle=False):
    """Prepare dataset example for training."""

    image = sample["image"].convert('RGB')
    iw, ih = image.size
    if use_predefined_cats:
        if 'prompt_close' in sample:
            prompt = sample['prompt_close']
        else:
            prompt = PROMPT_CLOSE_PSG if 'psg' in dataset_name else PROMPT_CLOSE_VG150
    else:
        prompt = PROMPT_SG

    use_think = 'think' in sample

    if remove_image_size_in_prompt:
        prompt = prompt.replace(f"of size ({iw} x {ih}) ", "")

    prompt = replace_answer_format(prompt)

    #normalize box to [0, 1000]
    objs = []
    for obj in json.loads(sample['objects']):
        box = obj['bbox']
        obj['bbox'] = [int(box[0]/iw*1000), int(box[1]/ih*1000),
                       int(box[2]/iw*1000), int(box[3]/ih*1000)]
        objs.append(obj)

    answer = format_answer(objs, sample["relationships"], shuffle=shuffle)
    if use_think:
        answer = '{}<answer>\n{}\n</answer>'.format(sample['think'], answer)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful and multimodal AI assistant."
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



@dataclass
class CustomScriptArguments(ScriptArguments):
    use_predefined_cats: bool = field(
        default=False, 
        metadata={"help": "Whether to use predefined object categories"}
    )



def main():
    accelerator = Accelerator()
    # args
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # load dataset 
    try:
        train_dataset = load_dataset(script_args.dataset_name)['train']
    except:
        train_dataset = load_from_disk(script_args.dataset_name)

    print(f"Training set size: {len(train_dataset)}")
    #print(f"Validation set size: {len(val_dataset)}")
    print("Train set[0]:", format_data(script_args.dataset_name, train_dataset[0], use_predefined_cats=script_args.use_predefined_cats))

    
    # model config.
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False, #if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    min_pixels = 3136
    max_pixels = 1024 * 28 * 28

    model_type=None
    base_name = None
    model_name = model_args.model_name_or_path.lower()

    if any(key in model_name for key in ['qwen2vl', 'qwen2-vl', 'qwen-2-vl']):
        model_type = "qwen2vl"
        if '7b' in model_name:
            base_name = "Qwen/Qwen2-VL-7B-Instruct"
        elif '2b' in model_name:
            base_name = "Qwen/Qwen2-VL-2B-Instruct"
        else:
            raise Exception(f"Unknown model size in: {model_name}")

    elif any(key in model_name for key in ['qwen2.5vl', 'qwen2.5-vl', 'qwen2-5-vl', 'qwen-2.5-vl']):
        model_type = "qwen2.5vl"
        if '7b' in model_name:
            base_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        elif '3b' in model_name:
            base_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        else:
            raise Exception(f"Unknown model size in: {model_name}")

    else:
        raise Exception(f"Unknown model type: {model_args.model_name_or_path}")

    processor = AutoProcessor.from_pretrained(base_name,
                    min_pixels=script_args.min_pixels,
                    max_pixels=script_args.max_pixels)
    model_cls = None
    if model_type == "qwen2vl":
        model_cls = Qwen2VLForConditionalGeneration
    elif model_type == "qwen2.5vl":
        model_cls = Qwen2_5_VLForConditionalGeneration

    assert model_cls is not None, " Unsupported model:{}".format(model_args.model_name_or_path)

    model = model_cls.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )

    class Collator(object):
        def __init__(self, dataset_name, processor, use_predefined_cats):
            self.dataset_name = dataset_name
            self.processor = processor
            self.use_predefined_cats = use_predefined_cats
            self._db = {}

        def __call__(self, examples):
            # Get the texts and images, and apply the chat template
            texts, image_inputs = [], []
            for example in examples:
                if str(example) not in self._db:
                    self._db[str(example)] = 0

                shuffle = (self._db[str(example)] > 0) & (random.random() > 0.5)
                format_example = format_data(self.dataset_name, example, use_predefined_cats=self.use_predefined_cats, shuffle=shuffle)['messages']
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
        data_collator=Collator(script_args.dataset_name, processor, script_args.use_predefined_cats),
        peft_config=get_peft_config(model_args),
    )
    # Check for existing checkpoint
    def find_valid_checkpoint(output_dir):
        ckpt_re = re.compile(r"checkpoint-(\d+)$")      # ↳ ends right after the digits
        
        checkpoints = sorted(
            [
                p for p in glob.glob(os.path.join(output_dir, "checkpoint-*"))
                if ckpt_re.search(os.path.basename(p))   # keep only pure-numeric checkpoints
            ],
            key=lambda p: int(ckpt_re.search(os.path.basename(p)).group(1))
        )
        for ckpt in reversed(checkpoints):  # Check latest first
            if glob.glob(os.path.join(ckpt, "global_step*")):
                return ckpt
        return None
    
    ckpt_to_resume = find_valid_checkpoint(training_args.output_dir)
    if ckpt_to_resume:
        print(f"[INFO] Resuming from checkpoint: {ckpt_to_resume}")
        trainer.train(resume_from_checkpoint=ckpt_to_resume)
    else:
        print("[INFO] Starting training from scratch")
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    main()
