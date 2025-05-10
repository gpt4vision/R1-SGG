#!/bin/bash


#SBATCH --job-name=7B_GH200_SFT_CLOSE_GRPO_lr2x
#SBATCH --time=12:00:00

#SBATCH --nodes=8  # 8 nodes, each has 4x GH200                   
#SBATCH --ntasks=8                   # Total tasks equals total nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288 # fixed for GH200

#SBATCH --account=a-a03
#SBATCH --partition=normal
#SBATCH --output=RL_gh200_%j_%N.out
#SBATCH --mail-user="zychen.uestc@gmail.com" --mail-type=ALL


set -x
# ---------- Environment Setup ----------
export NCCL_ASYNC_ERROR_HANDLING=1
export DEBUG_MODE=True
export WANDB_PROJECT=RL4SGG


GPUS_PER_NODE=4
GROUP_SIZE=8
MODEL_PATH=$1

DATA_PATH="JosephZ/vg150_train_sgg_prompt"
RUN_NAME="qwen2vl-7b-sft-grpo-g8-n1-bs32-close-lr6e-7-gh200"
export OUTPUT_DIR="${SCRATCH}/models/7B/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

export LOG_PATH=${OUTPUT_DIR}/debug.log
export STRICT_FORMAT=True

MAX_PIXELS=$((512 * 28 * 28))


MASTER_PORT=29500

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NUM_TRAIN_NODES=${#NODELIST[@]}
TRAIN_NODES_LIST=("${NODELIST[@]:0:$NUM_TRAIN_NODES}")


MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
echo "MASTER_ADDR: $MASTER_ADDR"



# GH200 has a very high bandwidth between CPU and GPU, we should use it!
# zero2:
# bsz_per_devie=16, OOM; Ok,  with CPU offload for optimizer, ~60h with 3x GPUs
# bsz_per_devie=8, 386s for 30 steps, ~60h with 3x GPUs
# bsz_per_devie=16, ~40h with 4x GPUs
#
#  batch size: PER DEVICE(16) * ACC(1) * GPU(4) * NODE(4) // GROUP_SIZE(8) = 32
TRAIN_CMD="open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATA_PATH} \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --custom_per_device_train_batch_size 8 \
    --deepspeed ./local_scripts/zero2_offload.json \
    --gradient_accumulation_steps 1 \
    --learning_rate 6e-7 \
    --logging_steps 1 \
    --use_vllm true \
    --use_local_vllm true\
    --bf16 true\
    --tf32 true\
    --report_to wandb \
    --gradient_checkpointing true \
    --max_pixels ${MAX_PIXELS} \
    --temperature 1 \
    --top_p 0.9 \
    --top_k 50 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --num_generations ${GROUP_SIZE} \
    --num_iterations 1 \
    --beta 0.0\
    --vllm_max_model_len 4096 \
    --vllm_gpu_memory_utilization 0.2 \
    --use_predefined_cats true \
    --ddp_timeout 3600 \
    --save_only_model false"

    
echo "start training with TRAIN_CMD=${TRAIN_CMD} ..."

srun torchrun --nnodes ${NUM_TRAIN_NODES} --nproc_per_node ${GPUS_PER_NODE} \
    --node_rank ${SLURM_NODEID} \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    ${TRAIN_CMD}
