#!/bin/bash

export HF_HOME=$SCRATCH/huggingface
# ---------- Environment Setup ----------
export NCCL_ASYNC_ERROR_HANDLING=1
export DEBUG_MODE=True
export WANDB_PROJECT=RL4SGG


GROUP_SIZE=8
MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
DATA_PATH="JosephZ/vg150_train_sgg_prompt"
RUN_NAME="qwen2vl-7b-grpo-g${GROUP_SIZE}-n1-gh200"
export OUTPUT_DIR="${SCRATCH}/models/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

MAX_PIXELS=$((512 * 28 * 28))



HEAD_NODE_IP=0.0.0.0
MASTER_PORT=29500



# GH200 has a very high bandwidth between CPU and GPU, we should use it!
# zero2:
# bsz_per_devie=16, OOM; Ok,  with CPU offload for optimizer, ~60h with 3x GPUs
# bsz_per_devie=8, 386s for 30 steps, ~60h with 3x GPUs
# bsz_per_devie=16, ~40h with 4x GPUs
TRAIN_CMD="open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATA_PATH} \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 16 \
    --deepspeed ./local_scripts/zero2.json \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --use_vllm true \
    --use_local_vllm true\
    --bf16 true\
    --tf32 true\
    --report_to wandb \
    --gradient_checkpointing true \
    --max_pixels ${MAX_PIXELS} \
    --temperature 0.3 \
    --top_p 0.001 \
    --top_k 1 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --num_generations ${GROUP_SIZE} \
    --num_iterations 1 \
    --beta 0.0\
    --use_liger_loss true\
    --vllm_max_model_len 4096 \
    --vllm_gpu_memory_utilization 0.25"

    
echo "start training..."
# Training: GPUs 0-3, batch size: 16*4//8=8
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=${HEAD_NODE_IP} \
    --master_port=${MASTER_PORT} \
    ${TRAIN_CMD} > debug-gh200.log 2>&1 &
