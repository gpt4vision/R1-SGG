import os
import json
import re
import torch
import glob
import argparse
from datasets import load_dataset
from transformers import AutoProcessor
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, GenerationConfig
from qwen_vl_utils import process_vision_info

import numpy as np
from PIL import Image, ImageDraw

from transformers import Qwen2_5_VLForConditionalGeneration

from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download

os.environ["NCCL_SOCKET_TIMEOUT"] = "3600000"  # 1 hours
os.environ["NCCL_BLOCKING_WAIT"] = "1"

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

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


def get_model(name, device_map="auto"):
    if "qwen2vl-7b" in name or "Qwen2-VL-7B" in name or "qwen2vl" in name: # hack
        print("Using model:", name)
        min_pixels = 4*28*28
        max_pixels = 1024*28*28
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", 
                                        min_pixels=min_pixels, max_pixels=max_pixels)

        try:
            local_model_path = snapshot_download(name)
            print(f"set model:{name} to local path:", local_model_path)
            name = local_model_path
        except:
            pass

        model = LLM(
            model=name, 
            limit_mm_per_prompt={"image": 2},
            dtype='bfloat16',
            device=device_map,
            max_model_len=8192,
        )
    else:
        raise Exception(f"Unknown model_id: {name}")

    return model, processor 



def format_data(sample):
    image = sample['image'].convert('RGB')
    iw, ih = image.size
    prompt = sample['prompt_open']
    #prompt = PROMPT2.format(width=iw, height=ih)

    def replace_answer_format(item: str) -> str:
        return item.replace("<answer>", "```").replace("</answer>", "```")

    messages = [
        {
            "role": "system",
            #"content": [{"type": "text", "text": "You are a helpful and multimodal AI assistant."}],
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": replace_answer_format(prompt)
                },
            ],
        },
    ]
    return {"messages": messages}

def parse_args():
    parser = argparse.ArgumentParser(description="Run model inference on a dataset.")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset identifier")
    parser.add_argument("--model_name", required=True, help="Model name to load")
    parser.add_argument("--output_dir", required=True, help="Directory to save the outputs")

    return parser.parse_args()

def main():
    # Parse command line arguments.
    args = parse_args()
    print("args:", args)

    # Initialize Accelerator for distributed training/inference.
    accelerator = Accelerator()
    local_rank = accelerator.local_process_index
    device = f"cuda:{local_rank}"  # each process occupies a GPU

    # Get rank and world size for manual splitting
    rank = torch.distributed.get_rank()  # GPU ID or node rank
    world_size = torch.distributed.get_world_size()  # Total number of GPUs/nodes


    # Load the model and processor.
    model, processor = get_model(args.model_name, device_map=device)
    sampling_params = SamplingParams(
        temperature=0.01,
        top_k=1,
        top_p=0.001,
        repetition_penalty=1.0,
        max_tokens=2048,
    )

    print(f"model_id: {args.model_name}", " generation_config:", sampling_params)

    class Collator(object):
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, examples):
            ids = [e['image_id'] for e in examples]
            gt_objs = [e['objects'] for e in examples]
            gt_rels = [e['relationships'] for e in examples]
    
            llm_inputs = []
            for example in examples:
                format_example = format_data(example)['messages']
                prompt = self.processor.apply_chat_template(format_example,
                                tokenize=False, add_generation_prompt=True)

                image_input = process_vision_info(format_example)[0]

                tmp = {"prompt": prompt, "multi_modal_data": {"image": image_input}}
                llm_inputs.append(tmp)

            return ids, gt_objs, gt_rels, llm_inputs



    # Load dataset from Hugging Face hub.
    dataset = load_dataset(args.dataset)['train']

    names = glob.glob(args.output_dir + "/*json")
    names = set([e.split('/')[-1].replace('.json', '') for e in tqdm(names)])
    ids = []
    for idx, item in enumerate(tqdm(dataset)):
        if item['image_id'] in names:
            continue
        ids.append(idx)
    dataset = dataset.select(ids)
    print("*"*100, " old:", len(names), " unhandled:", len(dataset))


    # Split dataset manually
    total_size = len(dataset)
    per_gpu_size = total_size // world_size
    start_idx = rank * per_gpu_size
    end_idx = total_size if rank == world_size - 1 else (rank + 1) * per_gpu_size
    
    subset = dataset.select(range(start_idx, end_idx))  # Select subset for this GPU
    print("*"*100, "\n rank:", rank, " world size:", world_size,
            "subset from", start_idx, " to ", end_idx, "\n", 
            "\n data[0]:", format_data(dataset[0]),
            "*"*100)

    data_loader = DataLoader(subset, batch_size=2, 
                             shuffle=False, 
                             collate_fn=Collator(processor),
                             pin_memory=True
                            )
    #data_loader = accelerator.prepare(data_loader)
    print(f"Local ID: {local_rank} | len(dataset): {len(data_loader)}")

    # Create output directory if it doesn't exist.
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Save to {args.output_dir}")

    # Iterate over the data loader.
    _iter = 0
    for (im_ids, gt_objs, gt_rels, batch) in tqdm(data_loader, desc=f"Progress at rank {local_rank}"):
        with torch.no_grad():
            # Now pass correctly formatted inputs to vLLM
            outputs = model.generate(batch, sampling_params=sampling_params)
            output_texts = [output.outputs[0].text for output in outputs]


        if local_rank == 0 and _iter % 100 == 0:
            print("*" * 100)
            print("nvidia-smi:")
            os.system("nvidia-smi")
            print("*" * 100)
            print("*"*100, "\n", "image_id:", im_ids[0], "\n", 
                  "Input:", batch[0]['prompt'], 
                  "Response:", output_texts[0], "\n",
                  "GT objs:", gt_objs[0], " GT rels.: ", gt_rels[0],
                    "*"*100)

        _iter += 1
        for im_id, gt_obj, gt_rel, output_text in zip(im_ids, gt_objs, gt_rels, output_texts):
            out = {"image_id": im_id, "response": output_text, 
                   "gt_objects": gt_obj, "gt_relationships": gt_rel 
                  }
            dst_file = os.path.join(args.output_dir, f"{im_id}.json")
            with open(dst_file, 'w') as fout:
                json.dump(out, fout)

    print("Rank:", rank, " finished!")
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    print("All jobs finished!")

if __name__ == "__main__":
    main()

