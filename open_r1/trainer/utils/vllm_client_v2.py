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
from typing import Optional, List

import torch
from torch import nn

from trl.import_utils import is_vllm_available
from .misc import is_requests_available


if is_requests_available():
    import requests
    from requests import ConnectionError

if is_vllm_available():
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup


logger = logging.getLogger(__name__)


class VLLMClient:
    """
    A client class to interact with a vLLM server.

    This class provides methods to generate completions, initialize and manage weight update groups, and update model
    weights in a distributed setting. Before using it, start the vLLM server with `trl vllm-serve`.

    Args:
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            IP address of the vLLM server.
        server_port (`int`, *optional*, defaults to `8000`):
            Port number of the vLLM server.
        group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group.
        connection_timeout (`float`, *optional*, defaults to `0.0`):
            Total timeout duration in seconds to wait for the server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.

    Examples:
        Run the vLLM server with the model `Qwen/Qwen2.5-7B`:

        ```
        $ trl vllm-serve --model Qwen/Qwen2.5-7B
        ...
        INFO:     Application startup complete.
        INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
        ```

        Use the client to generate completions and update model weights:

        ```python
        >>> from trl.extras.vllm_client import VLLMClient
        >>> client = VLLMClient()
        >>> client.generate(["Hello, AI!", "Tell me a joke"])
        [[2980, 498, 1492, 752, 448, 264, 13027, 8645, 30, 358, 2776, 4460, 311, 3270, 264, 2025],
         [911, 7988, 1251, 382, 3838, 653, 498, 1618, 4325, 879, 2581, 20027, 264, 21428, 30, 362]]

        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
        >>> client.update_model_params(model)
        ```
    """

    def __init__(
        self, hosts: List[str] = ["0.0.0.0"], server_port: int = 8000, group_port: int = 51216, connection_timeout: float = 0.0,
    ):
        if not is_requests_available():
            raise ImportError("requests is not installed. Please install it with `pip install requests`.")
        if not is_vllm_available():
            raise ImportError("vLLM is not installed. Please install it with `pip install vllm`.")

        self.sessions = [requests.Session() for _ in range(len(hosts))]
        self.hosts = hosts
        self.server_port = server_port
        self.group_port = group_port
        valid_hosts = []
        for host in hosts:
            status = self.check_server(host, connection_timeout)  # check server and fail after timeout
            if status:
                valid_hosts.append(host)
        self.hosts = valid_hosts
        self.init_communicator()
        atexit.register(self.close_communicator)  # when the client object is deleted, close the weight update group

        print(f"VLLMClient connected to server host={host}, port={server_port}")

    def check_server(self, host, total_timeout: float = 0.0, retry_interval: float = 2.0):
        """
        Check server availability with retries on failure, within a total timeout duration. If the server is not up
        after the total timeout duration, raise a `ConnectionError`.

        Args:
            retry_interval (`float`, *optional*, defaults to `2.0`):
                Interval in seconds between retries.
            total_timeout (`float`, *optional*, defaults to `0.0`):
                Total timeout duration in seconds.
        """
        url = f"http://{host}:{self.server_port}/health/"
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
                        f"The vLLM server can't be reached at {self.host}:{self.server_port} after {total_timeout} "
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

    def chat(
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
        total_hosts = len(self.hosts)
        chunk_size = max(1, len(prompts) // total_hosts)
        
        results = []
        for i, host in enumerate(self.hosts):
            if i == total_hosts - 1:
                batch = prompts[i*chunk_size:]
            else:
                batch = prompts[i*chunk_size : (i+1)*chunk_size]
                
            if len(batch) == 0:
                continue
            url = f"http://{host}:{self.server_port}/chat/"
            response = self.sessions[i].post(
                url,
                json={
                    "prompts": batch,
                    "n": n,
                    "repetition_penalty": repetition_penalty,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": min_p,
                    "max_tokens": max_tokens,
                    "guided_decoding_regex": guided_decoding_regex,
                },
            )
            if response.status_code == 200:
                output = response.json()["completion_ids"]
                results.extend(output)
            else:
                raise Exception(f"Request failed on host {host}: {response.status_code}, {response.text}")
        
        return results        


    def init_communicator(self):
        """
        Initializes the weight update group in a distributed setup for model synchronization.
        """
        for kk, host in enumerate(self.hosts):
            # Get the tensor parallel size from the server
            url = f"http://{host}:{self.server_port}/get_tensor_parallel_size/"
            response = requests.get(url)
            if response.status_code == 200:
                tensor_parallel_size = response.json()["tensor_parallel_size"]
            else:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")

            url = f"http://{host}:{self.server_port}/get_dp_world_size/"
            response = requests.get(url)
            if response.status_code == 200:
                dp_world_size = response.json()["dp_world_size"]
            else:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")

            world_size = tensor_parallel_size * dp_world_size + 1
            self.rank = world_size - 1  # The client's rank is the last process

            # Initialize weight update group
            url = f"http://{host}:{self.server_port}/init_communicator/"
            response = self.sessions[kk].post(url, json={"host": self.hosts[0], "port": self.group_port, "world_size": world_size})
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Set up the communication group for weight broadcasting
        pg = StatelessProcessGroup.create(host=self.hosts[0], port=self.group_port, rank=self.rank, world_size=world_size)
        self.pynccl_comm = PyNcclCommunicator(pg, device="cuda:0")

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the model and broadcasts it to other processes.

        Args:
            name (`str`):
                Name of the layer whose weights are being updated.
            weights (`torch.Tensor`):
                Tensor containing the updated weights.
        """
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        for kk, host in enumerate(self.hosts):
            url = f"http://{host}:{self.server_port}/update_named_param/"
            response = self.sessions[kk].post(url, json={"name": name, "dtype": dtype, "shape": shape})
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Broadcast the weights to the other processes
        self.pynccl_comm.broadcast(weights, src=self.rank, stream=torch.cuda.current_stream())
        self.pynccl_comm.group.barrier()

    def update_model_params(self, model: nn.Module):
        """
        Updates all parameters of the given model by calling `update_named_param` for each parameter in the model.

        Args:
            model (`nn.Module`):
                Model whose parameters (weights/biases) are to be updated.
        """
        for name, param in model.named_parameters():
            # Update each parameter individually
            print("name:", name, " norm:", param.data.norm())
            self.update_named_param(name, param.data)

    def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        for kk, host in enumerate(self.hosts):
            url = f"http://{host}:{self.server_port}/reset_prefix_cache/"
            response = self.sessions[kk].post(url)
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def close_communicator(self):
        """
        Closes the weight update group and cleans up the communication group.
        """
        for kk, host in enumerate(self.hosts):
            url = f"http://{host}:{self.server_port}/close_communicator/"
            response = self.sessions[kk].post(url)
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")


# Example usage
if __name__ == "__main__":
    from vllm import SamplingParams

    client = VLLMClient()

    # Generate completions
    responses = client.generate(["Hello, AI!", "Tell me a joke"], n=4, max_tokens=32, sampling_params=SamplingParams())
    print("Responses:", responses)  # noqa

    # Update model weights
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B").to("cuda")
    client.update_model_params(model)
