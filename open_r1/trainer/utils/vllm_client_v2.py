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

import atexit
import logging
import time
import copy
import json
from typing import Optional, List

import torch
from torch import nn
from huggingface_hub import snapshot_download

from trl.import_utils import is_vllm_available
from .misc import is_requests_available

import aiohttp
import asyncio
from collections import defaultdict

if is_requests_available():
    import requests
    from requests import ConnectionError


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import get_world_group
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.sampling_params import GuidedDecodingParams
    from vllm.worker.worker import Worker
else:
    Worker = object


logger = logging.getLogger(__name__)


class VLLMClient:
    """
    A client class to interact with a vLLM server for inference and distributed parameter updates.

    Args:
        hosts (List[str], optional): List of IPs of vLLM servers. Default is ["0.0.0.0"].
        server_ports (List[str], optional): List of ports of vLLM servers. Default is ['8000'].
        group_port (int, optional): Base port for weight synchronization. Default is 51216.
        client_rank (int, optional): Rank of this client in a distributed setup. Default is 0.
        connection_timeout (float, optional): Time in seconds to wait for server readiness. Default is 60.0.
    """    

    def __init__(self,
        hosts: List[str] = ["0.0.0.0"], 
        server_ports: List[str] = ['8000'], 
        group_port: int = 51216, 
        client_rank: int = 0,
        connection_timeout: float = 60.0,
        disable_weight_sync: bool = False,
        local_vllm: bool = False,
        model_name: str = None,
        revision: Optional[str] = None,
        tensor_parallel_size: int = 1, 
        pipeline_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.25, 
        dtype: str = "auto",
        enable_prefix_caching: Optional[bool] = None,
        max_model_len: int = 4096,
        limit_mm_per_prompt: int = 1,
        device: str="auto",
        log_file: Optional[str]=None,
        min_pixels: Optional[int] = 4*28*28,
        max_pixels: Optional[int] = 1024*28*28
    ):
        self.log_file = log_file
        # Set up the file logging redirection if log_file is provided.
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)
            # Optionally, if you wish to disable propagation to avoid duplicate logs,
            # uncomment the next line:
            # logger.propagate = False

        self.local_vllm = local_vllm
        self.llm = None
        self.disable_weight_sync = disable_weight_sync

        if self.local_vllm: # use vLLM on training nodes
            print("initialze vLLM inference on training nodes...")
            assert model_name, "model_name cannot be None!"
            try:
                local_model_path = snapshot_download(model_name)
                print(f"set model:{model_name} to local path:", local_model_path)
                model_name = local_model_path
            except Exception:
                pass            
            # 
            self.llm = LLM(
                model=model_name,
                revision=revision,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=dtype,
                enable_prefix_caching=enable_prefix_caching,
                max_model_len=max_model_len,
                limit_mm_per_prompt={"image": limit_mm_per_prompt},
                mm_processor_kwargs= {"max_pixels": max_pixels, "min_pixels": min_pixels},
                device=device
            )
        else:
            if not is_requests_available():
                raise ImportError("requests is not installed. Please install it with `pip install requests`.")
            if not is_vllm_available():
                raise ImportError("vLLM is not installed. Please install it with `pip install vllm`.")

            if isinstance(hosts, str):
                hosts = [h.strip() for h in hosts.split(',')]            
            if isinstance(server_ports, str):
                server_ports = [e.strip() for e in server_ports.split(',')]

            self.hosts = hosts
            self.server_ports = server_ports
            self.group_port = group_port

            # Create a new persistent event loop running in the background
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            # Initialize sessions and communicator
            self.sessions = self.loop.run_until_complete(self._create_sessions(hosts))
            self.connection_timeout = connection_timeout

            self.client_rank = client_rank  # build a client <-> server mapping. If we have 8 servers, we should create 8 clients.
            # check all servers
            for host, port in zip(self.hosts, self.server_ports):
                status = self.check_server(host, port, connection_timeout, 60.0)
                assert status, f"Failed to connect {host}: {port}"

            if not disable_weight_sync:
                self.loop.run_until_complete(self.init_communicator())
            
            atexit.register(self.cleanup)
            print(f"VLLMClient connected to server hosts={self.hosts}, port={self.server_ports}")        

    def cleanup(self):
        """Cleans up communicator and sessions on exit."""
        if not self.loop.is_closed():
            if (not self.disable_weight_sync) and (not self.local_vllm):
                self.loop.run_until_complete(self.close_communicator())
	
            if hasattr(self, "sessions"):	
            	self.loop.run_until_complete(self.close_sessions())

            self.loop.close()        

    async def _create_sessions(self, hosts: List[str]) -> List[aiohttp.ClientSession]:
        """
        Asynchronously creates aiohttp client sessions for the provided hosts.
        """
        sessions = []
        for _ in hosts:
            sessions.append(aiohttp.ClientSession())
        return sessions        

    def check_server(self, host, port, total_timeout: float = 0.0, retry_interval: float = 60.0):
        """
        Check server availability with retries on failure, within a total timeout duration. If the server is not up
        after the total timeout duration, raise a `ConnectionError`.

        Args:
            retry_interval (`float`, *optional*, defaults to `2.0`):
                Interval in seconds between retries.
            total_timeout (`float`, *optional*, defaults to `0.0`):
                Total timeout duration in seconds.
        """
        url = f"http://{host}:{port}/health/"
        start_time = time.time()  # Record the start time
        _cnt = 0
        while _cnt < 10:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    print(
                        f"The vLLM server can't be reached at {host}:{port} after {total_timeout} "
                        "seconds. Make sure the server is running by running `trl vllm-serve`."
                    )
            else:
                if response.status_code == 200:
                    logger.info("Server is up!")
                    return True

            # Retry logic: wait before trying again
            logger.info(f"Server is not up yet. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)
            _cnt += 1

        return False


    def run_chat(
        self,
        prompts: list[str],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
    ) -> list[list[str]]:
        """
        Unified wrapper for chat that handles sync (local_vllm) and async (remote vLLM) modes.
        """
        if self.local_vllm:
            if guided_decoding_regex is not None:
                guided_decoding = GuidedDecodingParams(backend="outlines", regex=guided_decoding_regex)
            else:
                guided_decoding = None

            sampling_params = SamplingParams(
                n=n,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_tokens,
                guided_decoding=guided_decoding,
            )
            # Use vLLM's chat interface
            all_outputs = self.llm.chat(prompts, sampling_params=sampling_params)
            results = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
            
            return results
        else:
            return self.loop.run_until_complete(
                self.chat(
                    prompts=prompts,
                    n=n,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_tokens,
                    guided_decoding_regex=guided_decoding_regex,
                )
            )

    async def chat(
        self,
        prompts: list[str],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
    ) -> list[list[str]]:
        """
        Generates model completions for the provided prompts.

        Args:
            prompts (`list[str]`):
                List of text prompts for which the model will generate completions.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each prompt.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.`1.0` means no truncation.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter. `-1` means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each prompt.
            guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
                Regular expression to guide the decoding process.

        Returns:
            `list[list[int]]`:
                List of lists of token IDs representing the model-generated completions for each prompt.
        """
        payload = {
            "prompts": [],
            "n": n,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "guided_decoding_regex": guided_decoding_regex,
        }
        tasks = []
        total_hosts = len(self.hosts)
        chunk_size = max(1, len(prompts) // total_hosts)

        for i, host in enumerate(self.hosts):
            session = self.sessions[i]
            start_idx = i * chunk_size
            end_idx = None if i == total_hosts - 1 else (i + 1) * chunk_size
            batch = prompts[start_idx:end_idx]
            if not batch:
                continue

            url = f"http://{host}:{self.server_ports[i]}/chat/"
            batch_payload = copy.deepcopy(payload)
            batch_payload["prompts"] = batch
            tasks.append(self._async_post(session, url, batch_payload))

        responses = await asyncio.gather(*tasks)
        results = []
        for res in responses:
            results.extend(res["completion_ids"])

        return results

    async def _async_post(self, session, url, payload):
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                raise Exception(f"Request failed: {response.status}, {await response.text()}")
            return await response.json()    


    async def init_communicator(self):
        """
        Initializes the weight update group in a distributed setup for model synchronization.
        """
        async def init_single_communicator(session, host, port):
            url_tp = f"http://{host}:{port}/get_tensor_parallel_size/"

            async with session.get(url_tp) as resp_tp:
                if resp_tp.status != 200:
                    raise Exception(f"Request failed: {resp_tp.status}, {await resp_tp.text()}")
                tensor_parallel_size = (await resp_tp.json())["tensor_parallel_size"]

            world_size = tensor_parallel_size + 1
            payload = {"host": host, 
                       "port": self.group_port + self.client_rank, 
                       "world_size": world_size,
                       "client_rank": self.client_rank
                       } # nccl group host is on the server side
            await self._async_post(session, f"http://{host}:{port}/init_communicator/", payload)
            return world_size

        tasks = [init_single_communicator(self.sessions[self.client_rank], self.hosts[self.client_rank], self.server_ports[self.client_rank])]
        results = await asyncio.gather(*tasks)

        # Use the world_size from the first result to set rank
        world_size = results[0]
        self.rank = world_size - 1            

        # Set up the communication group for weight broadcasting
        pg = StatelessProcessGroup.create(host=self.hosts[self.client_rank],
                                          port=self.group_port + self.client_rank, 
                                          rank=self.rank, world_size=world_size)
        self.pynccl_comm = PyNcclCommunicator(pg, device=torch.cuda.current_device())


    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the model and broadcasts it to other processes.
    
        Args:
            name (`str`): Name of the layer whose weights are being updated.
            weights (`torch.Tensor`): Tensor containing the updated weights.
        """
        if self.local_vllm: # directly load the weights
            self.update_local_llm(name, weights)
        else:
            dtype, shape = str(weights.dtype), tuple(weights.shape)
            payload = {"name": name, "dtype": dtype, "shape": shape}
    
            host, port = self.hosts[self.client_rank], self.server_ports[self.client_rank]
            url = f"http://{host}:{port}/update_named_param/"
            try:
                response = requests.post(url, json=payload, timeout=self.connection_timeout)
                if response.status_code != 200:
                    raise RuntimeError(f"[{host}] Failed to update param '{name}': {response.status_code} {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Failed to send update_named_param to {host}: {e}")
                raise        
    
            # Broadcast the weights to the other processes
            self.pynccl_comm.broadcast(weights, src=self.rank, stream=torch.cuda.current_stream())
            self.pynccl_comm.group.barrier()        

    def update_local_llm(self, name: str, weights: torch.Tensor):
        params = self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(weights=[(name, weights)])
        return params
        

    def update_model_params(self, model: nn.Module):
        """
        Updates all parameters of the given model by calling `update_named_param` for each parameter in the model.

        Args:
            model (`nn.Module`):
                Model whose parameters (weights/biases) are to be updated.
        """
        for name, param in model.named_parameters():
            self.update_named_param(name, param.data)        


    def update_model_in_chunks_from_named_list(self, named_params: list[tuple[str, torch.nn.Parameter]]):
        """
        Efficiently update model weights in dtype-wise chunks.

        """
        if self.local_vllm:
            updated_params = set()
            for name, param in named_params:
                param_tmp = self.update_local_llm(name, param.data)
                updated_params.update(param_tmp)
            return updated_params

        dtype_groups = defaultdict(list)
        for name, param in named_params:
            dtype_groups[str(param.dtype)].append((name, param))
    
        for dtype_str, group in dtype_groups.items():
            meta, tensors = [], []
            for name, param in group:
                meta.append({
                    "name": name,
                    "dtype": dtype_str,
                    "shape": list(param.shape)
                })
                tensors.append(param.data.contiguous().flatten())

            host, port = self.hosts[self.client_rank], self.server_ports[self.client_rank]

            url = f"http://{host}:{port}/load_chunked_params/"
            response = requests.post(url, json={"params": meta}, timeout=self.connection_timeout)
            assert response.status_code == 200
    
            buffer_tensor = torch.cat(tensors).to("cuda")
            self.pynccl_comm.broadcast(buffer_tensor, src=self.rank, stream=torch.cuda.current_stream())
            self.pynccl_comm.group.barrier()


    def reset_prefix_cache(self):
        """
        Synchronously resets the prefix cache on all vLLM servers.
        """
        if self.local_vllm:
            self.llm.llm_engine.reset_prefix_cache()
        else:
            host, port = self.hosts[self.client_rank], self.server_ports[self.client_rank]
            url = f"http://{host}:{port}/reset_prefix_cache/"
            try:
                response = requests.post(url, json={}, timeout=self.connection_timeout)
                if response.status_code != 200:
                    raise RuntimeError(f"[{host}] Failed to reset prefix cache: {response.status_code} {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Could not reset prefix cache on {host}: {e}")
                raise Exception(e)

    async def close_communicator(self):
        """
        Closes the weight update group and cleans up the communication group.
        """
        tasks = []
        host, port = self.hosts[self.client_rank], self.server_ports[self.client_rank]
        url = f"http://{host}:{port}/close_communicator/"

        tasks.append(self._async_post(self.sessions[self.client_rank], url, {}))
    
        await asyncio.gather(*tasks)        


    async def close_sessions(self):
        """
        Closes all aiohttp client sessions.
        """
        for session in self.sessions:
            await session.close()        

