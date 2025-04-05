#!/bin/bash

#SBATCH --job-name=GRPO_gh200
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4 # each has 4x GH200
#SBATCH --cpus-per-task=288

#SBATCH --account=a-a03
#SBATCH --partition=normal

#SBATCH --output=TrainVLLM_%j_%N.out
#SBATCH --mail-user="zychen.uestc@gmail.com" --mail-type=ALL

set -euo pipefail


export HF_HOME=$SCRATCH/huggingface
# ---------- Environment Setup ----------
export NCCL_ASYNC_ERROR_HANDLING=1
export DEBUG_MODE=True
export WANDB_PROJECT=RL4SGG


GROUP_SIZE=8
MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
DATA_PATH="JosephZ/vg150_train_sgg_prompt"
RUN_NAME="qwen2vl-7b-grpo-g${GROUP_SIZE}-n1-gh200"
OUTPUT_DIR="${SCRATCH}/models/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

TP_SIZE=1
PORT_BASE=8000
MAX_PIXELS=$((512 * 28 * 28))

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NUM_NODES=${#NODELIST[@]}


MIXED_NODES=1  # Set this dynamically if needed


HEAD_NODE=${NODELIST[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
RDZV_PORT=29500
IP_FILE="${OUTPUT_DIR}/ip_port_list.txt"
> "$IP_FILE"

for i in $(seq 0 $((MIXED_NODES - 1))); do
    node=${NODELIST[$i]}
    ip=$(srun --nodes=1 --ntasks=1 -w "$node" hostname --ip-address)
    echo "${ip}:$((PORT_BASE))" >> "$IP_FILE"
done

SERVER_IP=$(cut -d: -f1 $IP_FILE | paste -sd,)
SERVER_PORT=$(cut -d: -f2 $IP_FILE | paste -sd,)

TRAIN_CMD="open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATA_PATH} \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --use_vllm true \
    --vllm_server_host ${SERVER_IP} \
    --vllm_server_port ${SERVER_PORT} \
    --vllm_server_timeout 600 \
    --vllm_locate_same_node true\
    --vllm_locate_same_remain_gpus 4\
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --max_pixels ${MAX_PIXELS} \
    --temperature 0.3 \
    --top_p 0.001 \
    --top_k 1 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --num_generations 8 \
    --num_iterations 1 \
    --beta 0.0"

# ---------- Functions ----------
launch_mixed_node() {
    local i=$1
    local node=$2
    local log_file="${OUTPUT_DIR}/vllm_node_${i}_${node}.log"
    srun --nodes=1 --ntasks=1 -w "$node" bash -c "
        export RANK=$i
        export NODE_RANK=$i
        # vLLM: GPUs 3
        CUDA_VISIBLE_DEVICES=3 python src/vllm_server_v2.py \
            --model '${MODEL_PATH}' \
            --gpu_memory_utilization 0.9 \
            --enable-prefix-caching true \
            --dtype 'bfloat16' \
            --max_model_len 4096 \
            --tensor_parallel_size ${TP_SIZE} \
            --host '0.0.0.0' \
            --port ${PORT_BASE} > ${log_file} 2>&1 &

        # Training: GPUs 0-3
        CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nnodes ${NUM_NODES} --nproc_per_node 3 \
            --node_rank \$NODE_RANK \
            --rdzv_id grpo_run \
            --rdzv_backend c10d \
            --rdzv_endpoint ${HEAD_NODE_IP}:${RDZV_PORT} \
            ${TRAIN_CMD} &
        wait
    " &
}

launch_training_node() {
    local i=$1
    local node=$2
    srun --nodes=1 --ntasks=1 -w "$node" bash -c "
        export NODE_RANK=$i
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes ${NUM_NODES} --nproc_per_node 4 \
            --node_rank \$NODE_RANK \
            --rdzv_id grpo_run \
            --rdzv_backend c10d \
            --rdzv_endpoint ${HEAD_NODE_IP}:${RDZV_PORT} \
            ${TRAIN_CMD}
    " &
}

# ---------- Main Launcher ----------
for i in "${!NODELIST[@]}"; do
    if [ $i -lt ${MIXED_NODES} ]; then
        launch_mixed_node $i "${NODELIST[$i]}"
    else
        launch_training_node $i "${NODELIST[$i]}"
    fi
done

wait
