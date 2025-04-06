import argparse
import base64
from io import BytesIO
import json
import time
import random
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

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    prompts = []
    for kk, item in enumerate(tqdm(db)):
        if kk > 10: break
        prompt = prepare_messages(item)
        prompts.append(prompt)

    client = VLLMClient(
       local_vllm=True,
       model_name=args.model_name_or_path
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
    )    




    print("[INFO] Running vLLM inference...")
    t0 = time.time()
    prompts = [json.dumps(e) for e in prompts]
    print(len(prompts))

    generated_ids = client.run_chat(prompts, n=1, max_tokens=100,
                top_p=0.001, top_k=1, temperature=1.0)

    t1 = time.time() - t0
    #generated_ids = [torch.as_tensor(e) for e in generated_ids]
    outputs = processor.batch_decode(generated_ids, skip_special_tokens=True), 
    print(len(outputs))
    print("****** vLLM generated text:")
    for i in range(8):
        print(outputs[0][i])

    print(" cost:", t1)

    # check weight sync.
    llmp = client.llm.llm_engine.model_executor.driver_worker.model_runner.model
    llmp_dicts = dict(llmp.named_parameters())

    miss1 = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Fetch corresponding param from llmp
            llmp_param = llmp_dicts.get(name, None)
            if llmp_param is None:
                #print(f"[WARN] Parameter {name} not found in vLLM model.")
                  miss1.append( (name, param.data.min()) )
                  param.data += 100
                  continue
               
            # Compare tensors
            if not torch.allclose(param.data, llmp_param.data, atol=1e-5):
                print(f"[FAIL] Mismatch in param '{name}'")
            else:
                print(f"[PASS] Param '{name}' is synchronized.")
            param.data += 100


    print("\n", "*"*100, "start weight synchronization ...\n", "*"*100, "\n")
    max_chunk_size = 100 * 1024 * 1024  # 100 MB
    param_chunk = []
    current_chunk_size = 0     
    del llmp_dicts

    t0 = time.time()
    updated_params = set()
    for name, param in model.named_parameters():
        # Calculate the size of this parameter in bytes
        param_size = param.numel() * param.element_size()

        param_chunk.append((name, param.data))
        current_chunk_size += param_size
    
        # When the accumulated chunk reaches or exceeds 100MB, update the model parameters in one chunk.
        if current_chunk_size >= max_chunk_size:
            old = client.update_model_in_chunks_from_named_list(param_chunk)
            updated_params.update(old)
            # Reset for the next chunk
            param_chunk = []
            current_chunk_size = 0

    if param_chunk and client is not None:
        client.update_model_in_chunks_from_named_list(param_chunk)
    t1 = time.time()
    print("weight synchronization cost:", t1-t0)
    # check again
    llmp_dicts = dict(llmp.named_parameters())

    miss2 = []
    for name, param in model.named_parameters():
        # Fetch corresponding param from llmp
        llmp_param = llmp_dicts.get(name, None)
        if llmp_param is None:
            miss2.append((name, param.data.min()))
            #print(f"[WARN] Parameter {name} not found in vLLM model.")
            continue
        # Compare tensors
        if not torch.allclose(param.data, llmp_param.data, atol=1e-5):
            print(f"[FAIL] Mismatch in param '{name}'")
        else:
            print(f"[PASS] Param '{name}' is synchronized.")

    #import pdb; pdb.set_trace()
   
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
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Model ID or path.")
    args = parser.parse_args()
    main(args)
