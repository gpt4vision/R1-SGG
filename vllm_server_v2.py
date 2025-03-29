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

##########################################################
#
# for processing 128 prompts, vLLM.generate
# exp1: 128x rtx_4090 training + 1x4 4090 for vLLM: time cost ~70s
# exp2: 128x rtx_4090 training + 8x4 4090 for vLLM: time cost ~
##########################################################

import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, List, Dict

import io
import torch
import json

from trl import TrlParser

from open_r1.trainer.utils.misc import (
    is_fastapi_available,
    is_pydantic_available,
    is_uvicorn_available,
    is_vllm_available,
)
from huggingface_hub import snapshot_download


if is_fastapi_available():
    from fastapi import BackgroundTasks, FastAPI

if is_pydantic_available():
    from pydantic import BaseModel

if is_uvicorn_available():
    import uvicorn

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

# Use spawn method for CUDA multiprocessing
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class ChunkedParam(BaseModel):
    name: str
    dtype: str
    shape: List[int]

class ChunkedWeightsRequest(BaseModel):
    params: List[ChunkedParam]



class WeightSyncWorker(Worker):
    """
    A vLLM worker that enables weight synchronization between a client and multiple server workers.
    """

    def __init__(self, *args, **kwargs):
        if not is_vllm_available():
            raise ImportError(
                "vLLM is required to use the WeightSyncWorker. Please install it using `pip install vllm`."
            )
        super().__init__(*args, **kwargs)
        self.pynccl_comm = None  # Communicator for weight updates
        self.client_rank = None  # Source rank for broadcasting updated weights

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        if self.pynccl_comm is not None:
            raise RuntimeError("Weight update group already initialized. Call close_communicator first.")
        rank = get_world_group().rank
        # 
        pg = StatelessProcessGroup.create(host=host, port=port, 
                        rank=rank, 
                        world_size=world_size)

        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: torch.dtype, shape: Sequence[int]) -> None:
        if self.pynccl_comm is None:
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")
        weight = torch.empty(shape, dtype=dtype, device=self.device)
        self.pynccl_comm.broadcast(weight, src=self.client_rank, stream=torch.cuda.current_stream())
        self.pynccl_comm.group.barrier()

        self.model_runner.model.load_weights(weights=[(name, weight)])

    def load_chunked_params(self, param_meta: List[dict]):
        if self.pynccl_comm is None:
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

        # Group parameters by dtype
        dtype_groups: Dict[str, List[dict]] = {}
        for param in param_meta:
            dtype_groups.setdefault(param["dtype"], []).append(param)

        for dtype_str, group in dtype_groups.items():
            dtype = getattr(torch, dtype_str.split(".")[-1])
            total_elems = sum(torch.Size(p["shape"]).numel() for p in group)
            flat_tensor = torch.empty(total_elems, dtype=dtype, device=self.device)

            self.pynccl_comm.broadcast(flat_tensor, src=self.client_rank, stream=torch.cuda.current_stream())
            self.pynccl_comm.group.barrier()

            offset = 0
            for meta in group:
                numel = torch.Size(meta["shape"]).numel()
                chunk = flat_tensor[offset:offset + numel].view(meta["shape"])
                offset += numel
                print(os.getpid(), "received param:", meta, " chunk", chunk.shape) 
                self.model_runner.model.load_weights([(meta["name"], chunk)])


    def close_communicator(self) -> None:
        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None
            self.client_rank = None


@dataclass
class ScriptArguments:
    model: str = field(metadata={"help": "Model name or path to load the model from."})
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Revision to use for the model. If not specified, the default branch will be used."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    pipeline_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of pipeline parallel workers to use."},
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the type is determined based on the model configuration."
        },
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={"help": "Optional maximum model length to use for vLLM."},
    )
    limit_mm_per_prompt: Optional[int] = field(
        default=1,
        metadata={"help": "Limit on the maximum images per prompt."},
    )
    use_hf_model: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use HuggingFace model instead of vLLM."},
    )
    enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to enable prefix caching in vLLM."},
    )


def main(script_args: ScriptArguments):
    if not is_fastapi_available():
        raise ImportError(
            "FastAPI is required to run the vLLM serve script. Please install it using `pip install fastapi`."
        )
    if not is_pydantic_available():
        raise ImportError(
            "Pydantic is required to run the vLLM serve script. Please install it using `pip install pydantic`."
        )
    if not is_uvicorn_available():
        raise ImportError(
            "Uvicorn is required to run the vLLM serve script. Please install it using `pip install uvicorn`."
        )
    if not is_vllm_available():
        raise ImportError("vLLM is required to run the vLLM serve script. Please install it using `pip install vllm`.")

    try:
        local_model_path = snapshot_download(script_args.model)
        print(f"set model:{script_args.model} to local path:", local_model_path)
        model_name = local_model_path
    except Exception:
        model_name = script_args.model


    llm = LLM(
        model=model_name,
        revision=script_args.revision,
        tensor_parallel_size=script_args.tensor_parallel_size,
        pipeline_parallel_size=script_args.pipeline_parallel_size,
        gpu_memory_utilization=script_args.gpu_memory_utilization,
        dtype=script_args.dtype,
        enable_prefix_caching=script_args.enable_prefix_caching,
        max_model_len=script_args.max_model_len,
        worker_cls=WeightSyncWorker,
        limit_mm_per_prompt={"image": script_args.limit_mm_per_prompt},
    )
    app = FastAPI()
    print("*"*100, "\n Starting services...", "\n", "*"*100)

    @app.get("/health/")
    async def health():
        return {"status": "ok"}


    @app.get("/get_tensor_parallel_size/")
    async def get_tensor_parallel_size():
        return {"tensor_parallel_size": llm.llm_engine.parallel_config.tensor_parallel_size}


    class GenerateRequest(BaseModel):
        prompts: List[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None

    class GenerateResponse(BaseModel):
        completion_ids: List[List[int]]

    @app.post("/chat/", response_model=GenerateResponse)
    async def chat(request: GenerateRequest):
        # Guided decoding, if enabled
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)
        else:
            guided_decoding = None

        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            guided_decoding=guided_decoding,
        )

        # Use vLLM's chat interface
        all_outputs = llm.chat([json.loads(item) for item in request.prompts], sampling_params=sampling_params)
        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]

        return {"completion_ids": completion_ids}


    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int
        client_rank: int

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest, background_tasks: BackgroundTasks):
        total_workers = request.world_size

        background_tasks.add_task(
            llm.collective_rpc,
            "init_communicator",
            args=(request.host, request.port, total_workers),
        )
        return {"message": "Request received, initializing communicator from client rank:%s"%request.client_rank }

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: List[int]

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest, background_tasks: BackgroundTasks):
        dtype = getattr(torch, request.dtype.split(".")[-1])
        background_tasks.add_task(llm.collective_rpc, "update_named_param", args=(request.name, dtype, request.shape))
        return {"message": "Request received, updating named parameter"}

    @app.post("/load_chunked_params/")
    async def load_chunked_params(request: ChunkedWeightsRequest, background_tasks: BackgroundTasks):
        # Send metadata (not weights!) to all workers
        params = [param.dict() for param in request.params]
        background_tasks.add_task(
            llm.collective_rpc,
            "load_chunked_params",
            args=(params,)
        )
        return {"message": f"Request received to load {len(params)} parameters"}    

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        success = llm.llm_engine.reset_prefix_cache()
        return {"message": "Request received, resetting prefix cache status: " + str(success)}

    @app.post("/close_communicator/")
    async def close_communicator():
        llm.collective_rpc("close_communicator")
        return {"message": "Request received, closing communicator"}

    uvicorn.run(app, host=script_args.host, port=script_args.port)


def make_parser(subparsers: argparse._SubParsersAction = None):
    if subparsers is not None:
        parser = subparsers.add_parser("vllm-serve", help="Run the vLLM serve script", dataclass_types=ScriptArguments)
    else:
        parser = TrlParser(ScriptArguments)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args)
