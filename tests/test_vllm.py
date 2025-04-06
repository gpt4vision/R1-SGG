import argparse
import base64
from io import BytesIO
import json
import time
from typing import List
from transformers import AutoProcessor

import torch
from PIL import Image

from open_r1.trainer.utils.vllm_client_v2 import VLLMClient
from datasets import load_dataset
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration



def encode_image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def prepare_messages(item):
    image = item['image']
    prompt = item['prompt_open']
    encoded_image_text = encode_image_to_base64(image)
    base64_qwen = f"data:image/jpeg;base64,{encoded_image_text}"

    messages_vllm = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": base64_qwen}},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    return messages_vllm 




def main(args):
    db = load_dataset("JosephZ/vg150_val_sgg_prompt")['train']

    print(f"[INFO] Connecting to vLLM server at {args.hosts}:{args.server_port}")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    prompts = []
    for kk, item in enumerate(tqdm(db)):
        if kk > 10: break
        prompt = prepare_messages(item)
        prompts.append(prompt)


    client = VLLMClient(
        hosts=args.hosts, #.split(','),
        server_ports=args.server_port,
        group_port=args.group_port,
        connection_timeout=60,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )    
    for name, param in model.named_parameters():
        client.update_named_param(name, param)

    print("[INFO] Running vLLM inference...")
    t0 = time.time()
    prompts = [json.dumps(e) for e in prompts]
    print(len(prompts))

    generated_ids = client.loop.run_until_complete(client.chat(prompts, n=8, max_tokens=1024,
                top_p=0.001, top_k=1, temperature=0.01))

    t1 = time.time() - t0
    #generated_ids = [torch.as_tensor(e) for e in generated_ids]
    outputs = processor.batch_decode(generated_ids, skip_special_tokens=True), 
    print(len(outputs))
    print("****** vLLM generated text:", 
         outputs[0][0],
        " cost:", t1)



    def cal_cost(client, model, lens):
        cost = []
        for i in range(3):
            t0 = time.time()
            #client.update_model_in_chunks(model, lens)

            named_params = list(model.named_parameters())
            chunk_size = lens  # or tune based on memory
            
            for i in range(0, len(named_params), chunk_size):
                chunk = named_params[i:i+chunk_size]
                client.update_model_in_chunks_from_named_list(chunk)            
                
            t1 = time.time()
            cost.append(t1-t0)
        return sum(cost)/len(cost)

    def cal_cost_by_size(client, model, max_bytes):
        cost = []
        for i in range(3):
            t0 = time.time()
            chunks = []              # List to accumulate (name, param) tuples
            current_chunk_bytes = 0  # Accumulated memory size in bytes
    
            for name, param in model.named_parameters():
                param_bytes = param.numel() * param.element_size()
    
                # If adding this parameter would exceed the max_bytes limit
                if current_chunk_bytes + param_bytes > max_bytes:
                    # Process the current chunk if not empty
                    if chunks:
                        client.update_model_in_chunks_from_named_list(chunks)
                        chunks = []
                        current_chunk_bytes = 0
    
                # If the parameter itself exceeds max_bytes, process it individually
                if param_bytes > max_bytes:
                    client.update_model_in_chunks_from_named_list([(name, param)])
                else:
                    # Otherwise, add the parameter to the current chunk
                    chunks.append((name, param))
                    current_chunk_bytes += param_bytes
    
            # Process any remaining parameters
            if chunks:
                client.update_model_in_chunks_from_named_list(chunks)
    
            t1 = time.time()
            cost.append(t1 - t0)
        return sum(cost) / len(cost)    



    for k in range(1, 10):
        try:
            GB = (1<<30) * 0.1  * k
            print(f"update cost with chunk size={k} GB:", cal_cost_by_size(client, model, GB))
        except:
            print("Timeout at", k)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts", type=str, default="[127.0.0.1]", help="Host address of the vLLM server.")
    parser.add_argument("--server_port", type=str, default='8000', help="Port for vLLM API requests.")
    parser.add_argument("--group_port", type=int, default=51216, help="Port for NCCL communication.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Model ID or path.")
    args = parser.parse_args()
    main(args)
