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
import random
from PIL import Image, ImageDraw

from transformers import Qwen2_5_VLForConditionalGeneration

from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download

os.environ["NCCL_SOCKET_TIMEOUT"] = "3600000"  # 1 hours
os.environ["NCCL_BLOCKING_WAIT"] = "1"


from src.vg_synonyms import VG150_OBJ_CATEGORIES, VG150_PREDICATES

from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration



from open_r1.trainer.utils.misc import encode_image_to_base64, is_pil_image

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

from open_r1.trainer.utils.prompt_gallery import (
    PROMPT_SG,
    PROMPT_CLOSE_PSG,
    PROMPT_CLOSE_VG150,
    VG150_BASE_OBJ_CATEGORIES,
    VG150_BASE_PREDICATE,
    format_prompt_close_sg,
    VG150_OBJ_CATEGORIES,
    VG150_PREDICATES
)


def get_model(name, device_map="auto", max_model_len=4096):
    is_qwen2vl = 'qwen2vl' in name.lower() or 'qwen2-vl' in name.lower()
    is_qwen25vl = 'qwen2.5-vl' in name.lower() or 'qwen25-vl' in name.lower() or 'qwen2.5vl' in name.lower()
    is_llava = 'llava' in name.lower()
    base_model_name = None
    if is_qwen2vl or is_qwen25vl:
        print("Using model:", name)
        min_pixels = 4*28*28
        max_pixels = 1024*28*28
        if is_qwen2vl:
            if '7b' in name.lower():
                base_model_name = "Qwen/Qwen2-VL-7B-Instruct" 
            elif '2b' in name.lower():
                base_model_name = "Qwen/Qwen2-VL-2B-Instruct"
        if is_qwen25vl:
            if '7b' in name.lower():
                base_model_name = "Qwen/Qwen2.5-VL-7B-Instruct" 
            elif '3b' in name.lower():
                base_model_name = "Qwen/Qwen2.5-VL-3B-Instruct" 

        assert base_model_name is not None, "TODO: check the model -- {}".format(name)
        processor = AutoProcessor.from_pretrained(base_model_name, 
                                        min_pixels=min_pixels, max_pixels=max_pixels)

        try:
            local_model_path = snapshot_download(name)
            print(f"set model:{name} to local path:", local_model_path)
            name = local_model_path
        except:
            pass

        model = LLM(
            model=name, 
            limit_mm_per_prompt={"image": 1},
            dtype='bfloat16',
            device=device_map,
            max_model_len=max_model_len,
            mm_processor_kwargs= { "max_pixels": max_pixels, "min_pixels": min_pixels},
        )
    elif is_llava:
        model_cls = LlavaForConditionalGeneration if '1.5' in name else LlavaNextForConditionalGeneration
        model = model_cls.from_pretrained(
                  name, 
                  torch_dtype=torch.bfloat16, 
              ).to(device_map)
        processor = AutoProcessor.from_pretrained(name)
    else:
        raise Exception(f"Unknown model_id: {name}")

    return is_qwen2vl, is_qwen25vl, is_llava, model, processor 


def replace_answer_format(item: str) -> str:
    return item.replace("<answer>", "```json").replace("</answer>", "```")

def format_data(dataset_name, sample, use_predefined_cats=False, use_think_system_prompt=False, remove_image_size_in_prompt=True):
    image = sample['image'].convert('RGB')
    iw, ih = image.size
    if use_predefined_cats:
        prompt = PROMPT_CLOSE_PSG if 'psg' in dataset_name else PROMPT_CLOSE_VG150
    else:
        prompt = PROMPT_SG

    if remove_image_size_in_prompt:
        prompt = prompt.replace(f"of size ({iw} x {ih}) ", "")

    prompt = replace_answer_format(prompt)

    system_prompt = SYSTEM_PROMPT if use_think_system_prompt else "You are a helpful and multimodal AI assistant."

    base64_image = encode_image_to_base64(image)
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    return image, messages

def parse_args():
    parser = argparse.ArgumentParser(description="Run model inference on a dataset.")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset identifier")
    parser.add_argument("--model", required=True, help="Model name to load")
    parser.add_argument("--output_dir", required=True, help="Directory to save the outputs")
    parser.add_argument("--use_think_system_prompt", action="store_true", help="Use system prompt with <think>...</think>")
    parser.add_argument("--use_predefined_cats", action="store_true", help="Use predefined categories in the prompt")
    parser.add_argument("--max_model_len", type=int, default=4096, help="max_model_len for vLLM")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")

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
    is_qwen2vl, is_qwen25vl, is_llava, model, processor = get_model(args.model, device_map=device, max_model_len=args.max_model_len)
    sampling_params = SamplingParams(
        temperature=0.01,
        top_k=1,
        top_p=0.001,
        repetition_penalty=1.0,
        max_tokens=2048,
    )

    print(f"model_id: {args.model}", " generation_config:", sampling_params)

    class Collator(object):
        def __init__(self, data_name, 
                     processor, 
                     use_predefined_cats, use_think_system_prompt,
                     is_llava=False):
            self.data_name = data_name
            self.processor = processor
            self.use_predefined_cats = use_predefined_cats
            self.use_think_system_prompt = use_think_system_prompt
            self.is_llava = is_llava

        def __call__(self, examples):
            ids = [e['image_id'] for e in examples]
            gt_objs = [e['objects'] for e in examples]
            gt_rels = [e['relationships'] for e in examples]
    
            llm_inputs = []
            images = []
            for example in examples:
                image, prompt = format_data(self.data_name, example, 
                                     use_predefined_cats=self.use_predefined_cats, 
                                     use_think_system_prompt=self.use_think_system_prompt)

                if self.is_llava:
                    conversation = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt[-1]['content'][-1]['text']}, {"type": "image"},]}]
                    prompt_item = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                    llm_inputs.append(prompt_item)
                else:
                    llm_inputs.append(prompt)
                images.append(image)

            if self.is_llava:
                llm_inputs = self.processor(text=llm_inputs, images=images, padding=True, return_tensors="pt")
                input_height = input_width = [336]*len(images)
            else:
                texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                         for msg in llm_inputs]
                inputs = processor(
                    text=texts,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )
                input_height = [inputs['image_grid_thw'][idx][1].item()*14 for idx in range(len(images))]
                input_width = [inputs['image_grid_thw'][idx][2].item()*14 for idx in range(len(images))]      

            return ids, gt_objs, gt_rels, input_width, input_height, llm_inputs



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
            "\n data[0]:", format_data(args.dataset, dataset[0], use_predefined_cats=args.use_predefined_cats, use_think_system_prompt=args.use_think_system_prompt),
            "*"*100)

    data_loader = DataLoader(
                             subset, 
                             batch_size=args.batch_size, 
                             shuffle=False, 
                             collate_fn=Collator(args.dataset, processor, 
                                                 use_predefined_cats=args.use_predefined_cats, 
                                                 use_think_system_prompt=args.use_think_system_prompt,
                                                 is_llava=is_llava),
                             pin_memory=True
                            )
    #data_loader = accelerator.prepare(data_loader)
    print(f"Local ID: {local_rank} | len(dataset): {len(data_loader)}")

    # Create output directory if it doesn't exist.
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Save to {args.output_dir}")

    # Iterate over the data loader.
    _iter = 0
    for im_ids, gt_objs, gt_rels, input_width, input_height, batch in tqdm(data_loader, desc=f"Progress at rank {local_rank}"):
        with torch.no_grad():
            if is_llava:
                batch = batch.to(model.device)
                outputs = model.generate(**batch, max_new_tokens=2048)  
                output_texts = processor.batch_decode(outputs, skip_special_tokens=True)
                output_texts = [text.split("ASSISTANT:")[-1] for text in output_texts]
            else:
                outputs = model.chat(batch, sampling_params=sampling_params)
                output_texts = [output.outputs[0].text for output in outputs]

 
        if local_rank == 0 and _iter % 100 == 0:
            print("*" * 100)
            print("nvidia-smi:")
            os.system("nvidia-smi")
            print("*" * 100)
            print("*"*100, "\n", "image_id:", im_ids[0], "\n", 
                  "Response:", output_texts[0], "\n",
                  "GT objs:", gt_objs[0], " GT rels.: ", gt_rels[0],
                    "*"*100)

        _iter += 1
        for im_id, gt_obj, gt_rel, output_text, input_iw, input_ih in zip(im_ids, gt_objs, gt_rels, output_texts, input_width, input_height):
            if is_qwen2vl:
                box_scale = [1000.0, 1000.0]
            else:
                box_scale = [input_iw, input_ih]

            out = {"image_id": im_id, "response": output_text, 
                   "gt_objects": gt_obj, "gt_relationships": gt_rel,
                   "box_scale": box_scale
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
