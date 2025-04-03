#!/bin/bash

#SBATCH --job-name=GRPO_train_vllm
#SBATCH --time=24:00:00
#SBATCH --nodes=24
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=rtx_4090:8
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16000M 
#SBATCH --output=TrainVLLM_%j_%N.out
#SBATCH --mail-user="zychen.uestc@gmail.com" --mail-type=ALL

# ---------- Environment Setup ----------
export NCCL_ASYNC_ERROR_HANDLING=1
export DEBUG_MODE=True
export WANDB_PROJECT=RL4SGG

MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
DATA_PATH="JosephZ/vg150_train_sgg_prompt"
RUN_NAME="qwen2vl-7b-grpo-g8-n1"
OUTPUT_DIR="models/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

TP_SIZE=4
PORT_BASE=8000
MAX_PIXELS=$((512 * 28 * 28))

# Get all node hostnames
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NUM_NODES=${#NODELIST[@]}
GPUS_PER_NODE=8

# Head node setup
HEAD_NODE=${NODELIST[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
RDZV_PORT=29500

# ---------- Create IP:PORT File ----------
IP_FILE="${OUTPUT_DIR}/ip_port_list.txt"
> $IP_FILE

for node in "${NODELIST[@]}"; do
    ip=$(srun --nodes=1 --ntasks=1 -w "$node" hostname --ip-address)
    echo "${ip}:${PORT_BASE}" >> "$IP_FILE"
done

# Prepare comma-separated IPs and ports
SERVER_IP=$(cut -d: -f1 $IP_FILE | paste -sd,)
SERVER_PORT=$(cut -d: -f2 $IP_FILE | paste -sd,)

echo "SERVER_IP=$SERVER_IP"
echo "SERVER_PORT=$SERVER_PORT"
echo "HEAD_NODE_IP=$HEAD_NODE_IP"


# allocate 24 nodes, 2/3 for training, 1/3 for vllm inference
# 
#      node-0      node-1          node-14        node-15    ....
# GPU [0,1,2,3] [0, 1, 2, 3] ... [0, 1, 2, 3] [0, 1, 2, 3] [0-7] [0-7] [0-7]  # 8*8+ 4*16=128
# GPU [4,5,6,7] [4, 5, 6, 7] ... [4, 5, 6, 7] [4, 5, 6, 7] -> 64


# ---------- Launch on Each Node ----------
for i in "${!NODELIST[@]}"; do
    node=${NODELIST[$i]}
    VLLM_LOG="${OUTPUT_DIR}/vllm_node_${i}_${node}.log"

    srun --nodes=1 --ntasks=1 -w $node bash -c "
        # ------------------- Launch vLLM on GPUs 4-7 -------------------
        (
            export RANK=$i
            export CUDA_VISIBLE_DEVICES=4,5,6,7
            echo \"Launching vLLM on $node (GPUs 4-7)\"

            python src/vllm_server_v2.py \
                --model '${MODEL_PATH}' \
                --gpu_memory_utilization 0.85 \
                --enable-prefix-caching true \
                --dtype 'bfloat16' \
                --max_model_len 4096 \
                --tensor_parallel_size ${TP_SIZE} \
                --host '0.0.0.0' \
                --port ${PORT_BASE} > ${VLLM_LOG} 2>&1 
        ) &

        # ------------------- Launch Training on GPUs 0-3 -------------------
        (
            export NODE_RANK=$i
            export CUDA_VISIBLE_DEVICES=0,1,2,3
            echo \"Launching training on $node (GPUs 0-3)\"

            torchrun --nnodes ${NUM_NODES} --nproc_per_node 4 \
                --node_rank \$NODE_RANK \
                --rdzv_id grpo_run \
                --rdzv_backend c10d \
                --rdzv_endpoint ${HEAD_NODE_IP}:${RDZV_PORT} \
                open_r1/grpo.py \
                --output_dir ${OUTPUT_DIR} \
                --model_name_or_path ${MODEL_PATH} \
                --dataset_name ${DATA_PATH} \
                --deepspeed ./local_scripts/zero3.json \
                --max_prompt_length 2048 \
                --max_completion_length 1024 \
                --per_device_train_batch_size 1 \
                --gradient_accumulation_steps 2 \
                --logging_steps 1 \
                --use_vllm true \
                --vllm_server_host ${SERVER_IP} \
                --vllm_server_port ${SERVER_PORT} \
                --vllm_server_timeout 600 \
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
		--beta 0.0 
        ) &

        wait
    " &
done

wait
