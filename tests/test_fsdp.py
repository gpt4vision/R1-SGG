import torch
import torch.nn as nn
import torch.distributed as dist
import os
import time
from io import BytesIO
import base64
import json
from contextlib import nullcontext

import deepspeed
from accelerate.utils import DistributedType
from accelerate import Accelerator
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration

from open_r1.trainer.utils.vllm_client_v2 import VLLMClient
from datasets import load_dataset


def encode_image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def prepare_messages(image):
    encoded_image_text = encode_image_to_base64(image)
    base64_qwen = f"data:image/jpeg;base64,{encoded_image_text}"

    messages_vllm = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": base64_qwen}},
                {"type": "text", "text": "Describe this image."},
            ],
        },
    ]

    return messages_vllm



def main():
    model_name = "Qwen/Qwen2-VL-7B-Instruct"

    accelerator = Accelerator()
    device = accelerator.device

    processor = Qwen2VLProcessor.from_pretrained(model_name, max_pixels=512*28*28)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                attn_implementation='flash_attention_2'
            )
    model = accelerator.prepare_model(model)


    """ test clients """
    def get_gateway_client_id(world_size, rank, gpus_per_node, num_clients):
        num_nodes = world_size // gpus_per_node
        client_ranks = [
            (i % num_nodes) * gpus_per_node + (i // num_nodes)
            for i in range(num_clients)
        ]
        if rank in client_ranks:
            return client_ranks.index(rank)
        return None

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    gpus_per_node = torch.cuda.device_count()


    hosts, ports = [], []
    for line in open("ip_port_test.txt"):
        host, port =line.strip().split(':')
        hosts.append(host.strip())
        ports.append(port.strip())    

    num_clients = len(hosts)
    client_id = get_gateway_client_id(world_size, rank, gpus_per_node, num_clients)

    # create N=len(hosts) clients
    if client_id is not None:
        vllm_client = VLLMClient(
            hosts, ports,
            connection_timeout=360,
            client_rank = client_id
        )
        print("*"*100, "\n Create VLLMClient at rank:", rank, " cliend_rank:", client_id)
    else:
        vllm_client = None    


    """ test chat """

    if accelerator.is_main_process:

        db = load_dataset("JosephZ/vg150_val_sgg_prompt")['train']
        prompts = []
        for kk, item in enumerate(tqdm(db)):
            if len(prompts) >=128: break
            prompt = prepare_messages(item['image'])
            prompts.append(prompt)    

        print("[INFO] Running vLLM inference...")
        t0 = time.time()
        prompts = [json.dumps(e) for e in prompts]
        print(len(prompts))

        generated_ids = vllm_client.loop.run_until_complete(vllm_client.chat(prompts, n=1, max_tokens=50,
                    top_p=0.001, top_k=1, temperature=0.01))

        t1 = time.time() - t0
        #generated_ids = [torch.as_tensor(e) for e in generated_ids]
        outputs = processor.batch_decode(generated_ids, skip_special_tokens=True),
        print(len(outputs))
        print("****** vLLM generated text:",
             outputs,
            " cost:", t1)

    """ test weight synchronization """
    max_chunk_size = 100 * 1024 * 1024  # 100 MB
    param_chunk = []
    current_chunk_size = 0
    debug_file = "tests/debug_%s.log" % accelerator.process_index
    with open(debug_file, 'w') as fout:
        pass

    is_fsdp_used = accelerator.distributed_type == DistributedType.FSDP
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3    
    gather_if_zero3 = deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext    

    if is_fsdp_used:
        print("*"*100, "\n Test FSDP ...\n", "*"*100)
        with FSDP.summon_full_params(model, recurse=True, writeback=False):
            for name, param in model.named_parameters():
                if param.data is None:
                    continue
                if vllm_client is not None:
                    # Calculate the size of this parameter in bytes
                    param_size = param.numel() * param.element_size()

                    param_chunk.append((name, param.data))
                    current_chunk_size += param_size

                    # When the accumulated chunk reaches or exceeds 100MB, update the model parameters in one chunk.
                    if current_chunk_size >= max_chunk_size:
                        if os.path.exists(debug_file):
                            with open(debug_file, 'a') as fout:
                                names = [(p[0], p[1].shape) for p in param_chunk]
                                cmd = f"FSDP --- rank={accelerator.process_index}, send params={names}\n"
                                fout.write(cmd)
                        vllm_client.update_model_in_chunks_from_named_list(param_chunk)
                        # Reset for the next chunk
                        param_chunk = []
                        current_chunk_size = 0
    else:
        print("*"*100, "\n Test non-FSDP ...\n", "*"*100)
        for name, param in self.model.named_parameters():
            with gather_if_zero3([param]): # gather if zero3 used
                if vllm_client is not None:
                    # Calculate the size of this parameter in bytes
                    param_size = param.numel() * param.element_size()

                    param_chunk.append((name, param.data))
                    current_chunk_size += param_size

                    # When the accumulated chunk reaches or exceeds 100MB, update the model parameters in one chunk.
                    if current_chunk_size >= max_chunk_size:
                        if os.path.exists(debug_file):
                            with open(debug_file, 'a') as fout:
                                names = [(p[0], p[1].shape) for p in param_chunk]
                                cmd = f"rank={accelerator.process_index}, send params={names}\n"
                                fout.write(cmd)
                        vllm_client.update_model_in_chunks_from_named_list(param_chunk)
                        # Reset for the next chunk
                        param_chunk = []
                        current_chunk_size = 0

    # If any parameters remain that didn't reach the 100MB threshold, update them as well.
    if param_chunk and vllm_client is not None:
        if os.path.exists(debug_file):
            with open(debug_file, 'a') as fout:
                names = [(p[0], p[1].shape) for p in param_chunk]
                cmd = f"rank={accelerator.process_index}, send params={names}\n"
                fout.write(cmd)
        vllm_client.update_model_in_chunks_from_named_list(param_chunk)





if __name__ == "__main__":
    main()
