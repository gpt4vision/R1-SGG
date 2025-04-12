#!/bin/bash


#SBATCH --job-name=GRPO_train_A100
#SBATCH --time=24:00:00

#SBATCH --nodes=4  # 4 nodes, each has 4x A100  
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128

#SBATCH --account=a-a03
#SBATCH --partition=normal
#SBATCH --output=RL_A100_%j_%N.out
#SBATCH --mail-user="zychen.uestc@gmail.com" --mail-type=ALL


# ---------- Environment Setup ----------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export NCCL_ASYNC_ERROR_HANDLING=1
export DEBUG_MODE=True
export WANDB_PROJECT=RL4SGG


TP_SIZE=4
GPUS_PER_NODE=4
GROUP_SIZE=8
MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
DATA_PATH="JosephZ/vg150_train_sgg_prompt"
RUN_NAME="qwen2vl-7b-grpo-g${GROUP_SIZE}-n1-temp1-topk50-top0.9"
export OUTPUT_DIR="${SCRATCH}/models/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

MAX_PIXELS=$((512 * 28 * 28))


MASTER_PORT=29500

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NUM_TRAIN_NODES=${#NODELIST[@]}
TRAIN_NODES_LIST=("${NODELIST[@]:0:$NUM_TRAIN_NODES}")

# Choose the first training node as the rendezvous head node
HEAD_NODE=${TRAIN_NODES_LIST[0]}

#MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)

MASTER_ADDR=$(echo "${SLURM_NODELIST}" | sed 's/[],].*//g; s/\[//g')
echo "MASTER_ADDR: $MASTER_ADDR"


PORT_BASE=8000
IP_FILE="${OUTPUT_DIR}/ip_port_list.txt"
> "$IP_FILE"

for i in $(seq 0 $((MIXED_NODES - 1))); do
    node=${NODELIST[$i]}
    ip=$(srun --nodes=1 --ntasks=1 -w "$node" hostname --ip-address)
    echo "${ip}:$((PORT_BASE))" >> "$IP_FILE"
done

SERVER_IP=$(cut -d: -f1 $IP_FILE | paste -sd,)
SERVER_PORT=$(cut -d: -f2 $IP_FILE | paste -sd,)



# batch size: per_device(2)*GPUS(4)*NODES(2)*ACC(8) //8=16
# local vLLM: 80G*0.25=20G
#
TRAIN_CMD="open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATA_PATH} \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 2 \
    --deepspeed ./local_scripts/zero2.json \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --use_vllm true \
    --vllm_collocate true \
    --vllm_server_host ${SERVER_IP} \
    --vllm_server_port ${SERVER_PORT} \
    --vllm_server_timeout 600 \
    --vllm_tp_size ${TP_SIZE} \
    --bf16 true\
    --tf32 true\
    --report_to wandb \
    --gradient_checkpointing true \
    --max_pixels ${MAX_PIXELS} \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --num_generations ${GROUP_SIZE} \
    --num_iterations 1 \
    --beta 0.0\
    --vllm_max_model_len 4096 \
    --vllm_gpu_memory_utilization 0.25\
    --save_only_model false"

    
echo "start training..."



launch_mixed_node() {
    local i=$1
    local node=$2
    local log_file="${OUTPUT_DIR}/vllm_node_${i}_${node}.log"
    srun --nodes=1 --ntasks=1 -w "$node" bash -c "
        export RANK=$i
        export NODE_RANK=$i
        CUDA_VISIBLE_DEVICES=0,1,2,3 python src/vllm_server_v2.py \
            --model '${MODEL_PATH}' \
            --gpu_memory_utilization 0.25 \
            --enable-prefix-caching true \
            --dtype 'bfloat16' \
            --max_model_len 4096 \
            --tensor_parallel_size ${TP_SIZE} \
            --host '0.0.0.0' \
            --port ${PORT_BASE} > ${log_file} 2>&1 &

        # Training: GPUs 0-3
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes ${NUM_TRAIN_NODES} --nproc_per_node 4 \
            --node_rank \$NODE_RANK \
            --rdzv_id grpo_run \
            --rdzv_backend c10d \
            --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
            ${TRAIN_CMD} &
        wait
    " &
}

# ---------- Main Launcher ----------
for i in "${!NODELIST[@]}"; do
    launch_mixed_node $i "${NODELIST[$i]}"
done

wait
