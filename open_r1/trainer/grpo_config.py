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

import warnings
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class GRPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`GRPOTrainer`].

    Only the parameters specific to GRPO training are listed here. For details on other parameters, refer to the
    [`~transformers.TrainingArguments`] documentation.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`GRPOTrainer`] is provided as a string.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
        num_generations (`int` or `None`, *optional*, defaults to `8`):
            Number of generations per prompt to sample. The global batch size (num_processes * per_device_batch_size)
            must be divisible by this value.
        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum length of the generated completion.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.

        > Parameters that control generation

        temperature (`float`, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int` or `None`, *optional*, defaults to `50`):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, top-k-filtering is
            disabled.
        min_p (`float` or `None`, *optional*, defaults to `None`):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between `0.0` and `1.0`. Typical values are in the `0.01-0.2` range.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat
            tokens.
        cache_implementation (`str` or `None`, *optional*, defaults to `None`):
            Implementation of the cache method for faster generation when use_vllm is set to False.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept unused for
            training, as vLLM will require one for generation. vLLM must be installed (`pip install vllm`).
        vllm_server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the vLLM server to connect to.
        vllm_server_port (`str`, *optional*, defaults to `8000`):
            Port of the vLLM server to connect to.
        vllm_server_timeout (`float`, *optional*, defaults to `120.0`):
            Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.
        vllm_guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
            Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled.

        > Parameters that control the training

        learning_rate (`float`, *optional*, defaults to `1e-6`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
        beta (`float`, *optional*, defaults to `0.04`):
            KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving training
            speed, but may be numerically unstable for long training runs.
        num_iterations (`int`, *optional*, defaults to `1`):
            Number of iterations per batch (denoted as μ in the algorithm).
        epsilon (`float`, *optional*, defaults to `0.2`):
            Epsilon value for clipping.
        epsilon_high (`float` or `None`, *optional*, defaults to `None`):
            Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound
            specified in argument `epsilon`. Paper [DAPO](https://huggingface.co/papers/2503.14476) recommends `0.28`.
        reward_weights (`list[float]` or `None`, *optional*, defaults to `None`):
            Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are
            weighted equally with weight `1.0`.
        scale_rewards (`bool`, *optional*, defaults to `True`):
            Whether to scale the rewards by dividing them by their standard deviation. If `True` (default), the rewards
            are normalized by the standard deviation, ensuring they have unit variance. If `False`, no scaling is
            applied. The [Dr. GRPO](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf)
            paper recommends not scaling the rewards, as scaling by the standard deviation introduces a question-level
            difficulty bias.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using
            the `ref_model_mixup_alpha` parameter. This synchronization originites from the
            [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        ref_model_mixup_alpha (`float`, *optional*, defaults to `0.6`):
            α parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which controls the mix
            between the current policy and the previous reference policy during updates. The reference policy is
            updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you
            must set `sync_ref_model=True`.
        ref_model_sync_steps (`int`, *optional*, defaults to `512`):
            τ parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which determines how
            frequently the current policy is synchronized with the reference policy. To use this parameter, you must
            set `sync_ref_model=True`.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is
            installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`.
    """

    # Parameters that control the model and reference model
    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `GRPOTrainer` is provided as a string."
        },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The global batch size (num_processes * per_device_batch_size) "
            "must be divisible by this value."
        },
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option "
            "is not compatible with vLLM generation."
        },
    )

    # Parameters that control generation
    temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled."
        },
    )
    min_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
        },
    )
    cache_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Implementation of the cache method for faster generation when use_vllm is set to False."},
    )
    use_liger_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the Liger GRPO loss."},
    )
    custom_per_device_train_batch_size: int = field(
        default=-1, 
        metadata={"help": "customized per_device_train_batch_size"}
    )
    keep_top_group_ratio: float = field(
        default=1,
        metadata={"help": "the ratio of keeping groups with top reward std."}
    )
    # Parameters that control generation acceleration powered by vLLM
    use_vllm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, ensure that a vLLM server is "
            "running. To run the server, install vLLM (`pip install vllm`) and run `trl vllm-serve`."
        },
    )
    use_local_vllm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use local vLLM engine."
        },
    )
    vllm_locate_same_node: bool = field(
        default=True,
        metadata={
            "help": "Whether to allocate vLLM and training jobs on same nodes."
            },
    )
    vllm_locate_same_remain_gpus: int = field(
        default=4,
        metadata={
            "help": "the number of remain GPUs for training if vLLM servers also be allocated on same nodes."
            }
    )
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host of the vLLM server to connect to."},
    )
    vllm_server_port: str = field(
        default='8000',
        metadata={"help": "Port of the vLLM server to connect to."},
    )
    vllm_server_timeout: float = field(
        default=120.0,
        metadata={
            "help": "Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up "
            "after the timeout, a `ConnectionError` is raised."
        },
    )
    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={"help": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."},
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    epsilon_high: Optional[float] = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
            "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`."
        },
    )
    reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )
    reward_warmup_steps: Optional[list[int]] = field(
        default=None,
        metadata={"help": "warmup steps for rewards."}
    )
    split_rewards: bool = field(
        default=False,
        metadata={"help": "whether to split the rewards"}
    )
    scale_rewards: bool = field(
        default=True,
        metadata={
            "help": "Whether to scale the rewards by dividing them by their standard deviation. If `True` (default), "
            "the rewards are normalized by the standard deviation, ensuring they have unit variance. If `False`, no "
            "scaling is applied. The Dr. GRPO paper recommends not scaling the rewards, as scaling by the standard "
            "deviation introduces a question-level difficulty bias."
        },
    )
    loss_type: str = field(
        default="bnpo",
        metadata={
            "help": "Specifies the loss formulation to use. Supported values are `grpo`, `bnpo`, and `dr_grpo`. "
            "`'grpo'`: Aggregates token-level losses by normalizing over sequence length. Not recommended due to "
            "length bias—this approach tends to prefer shorter completions with positive advantages and longer ones "
            "with negative advantages. "
            "`'bnpo'`: Aggregates token-level losses by normalizing number of active token in the local batch. "
            "Note that normalization is performed over the local batch only, so results may slightly vary depending "
            "on the local batch size, despite a constant effective batch size. When using "
            "`per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss. "
            "`'dr_grpo'`: Aggregates token-level losses by normalizing with a global constant. This method was "
            "introduced in the Dr. GRPO paper to eliminate length bias. The value of the constant corresponds to "
            "`max_completion_length`."
        },
    )
    move_ref_model_to_vllm: bool = field(
        default=False,
        metadata={
            "help": "Whether to move the reference model to a remote vLLM server. This can reduce GPU memory if sync_ref_model=False."
            }
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.6,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "
            "previous reference policy during updates. The reference policy is updated according to the equation: "
            "`π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: int = field(
        default=512,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is "
            "synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`."
        },
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."
        },
    )
    vllm_log_file: Optional[str] = field(
        default=None,
        metadata={"help": "log_file of vllm servers"}
    )
    # Deprecated parameters
    vllm_device: Optional[str] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.18.0. To use vLLM, start a vLLM "
            "server with the `trl vllm-serve` command."
        },
    )
    vllm_gpu_memory_utilization: Optional[float] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.18.0. To control the GPU memory "
            "utilization for vLLM, you should now use the `gpu_memory_utilization` parameter in the vLLM server "
            "configuration."
        },
    )
    vllm_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.18.0. To control the data type for "
            "vLLM generation, you should now use the `dtype` parameter in the vLLM server configuration."
        },
    )
    vllm_max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.18.0. To control the "
            "`max_model_len` for vLLM, you should now use the `max_model_len` parameter in the vLLM server "
            "configuration."
        },
    )
    vllm_enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.18.0. To control prefix caching in "
            "vLLM, you should now use the `enable_prefix_caching` parameter in the vLLM server configuration."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        if self.vllm_device is not None:
            warnings.warn(
                "`vllm_device` is deprecated and will be removed in version 0.18.0. To use vLLM, start a vLLM server "
                "with the `trl vllm-serve` command.",
                DeprecationWarning,
            )

        if self.vllm_gpu_memory_utilization is not None:
            warnings.warn(
                "`vllm_gpu_memory_utilization` is deprecated and will be removed in v0.18. To control the GPU memory "
                "utilization for vLLM, you should now use the `gpu_memory_utilization` parameter in the vLLM server "
                "configuration.",
                DeprecationWarning,
            )

        if self.vllm_dtype is not None:
            warnings.warn(
                "`vllm_dtype` is deprecated and will be removed in version 0.18.0. To control the data type for vLLM "
                "generation, you should now use the `dtype` parameter in the vLLM server configuration.",
                DeprecationWarning,
            )

        if self.vllm_max_model_len is not None:
            warnings.warn(
                "`vllm_max_model_len` is deprecated and will be removed in version 0.18.0. To control the "
                "`max_model_len` for vLLM, you should now use the `max_model_len` parameter in the vLLM server "
                "configuration.",
                DeprecationWarning,
            )

        if self.vllm_enable_prefix_caching is not None:
            warnings.warn(
                "`vllm_enable_prefix_caching` is deprecated and will be removed in version 0.18.0. To control prefix "
                "caching in vLLM, you should now use the `enable_prefix_caching` parameter in the vLLM server "
                "configuration.",
                DeprecationWarning,
            )
