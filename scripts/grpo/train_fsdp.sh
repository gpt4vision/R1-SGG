#!/bin/bash



#SBATCH --job-name=GRPO_train
#SBATCH --time=24:00:00
#SBATCH --nodes=16                   # 4 training nodes + 1 vLLM node = 5 nodes
#SBATCH --ntasks=16                   # Total tasks equals total nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=rtx_4090:8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=25000M
#SBATCH --output=RL_%j_%N.out
#SBATCH --mail-user="zychen.uestc@gmail.com" --mail-type=ALL


# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
#export NCCL_IB_DISABLE=1

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

# wait for vLLM servers
#sleep 60

# Read IPs from file and join them with commas
#ip_str=$(paste -sd, ip_list.txt)
#echo "vLLM servers: $ip_str"

FILE="ip_port_list.txt"

SERVER_IP=""
SERVER_PORT=""

while IFS=: read -r ip port; do
    SERVER_IP+="${ip},"
    SERVER_PORT+="${port},"
done < "$FILE"

# Remove trailing commas
SERVER_IP="${SERVER_IP%,}"
SERVER_PORT="${SERVER_PORT%,}"

echo "SERVER_IP=$SERVER_IP"
echo "SERVER_PORT=$SERVER_PORT"


# Define node counts
NUM_TRAIN_NODES=${SLURM_NNODES} # all nodes
GPUS_PER_NODE=8

# Get the list of allocated nodes
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# Assign training nodes (first NUM_TRAIN_NODES nodes)
TRAIN_NODES=("${NODELIST[@]:0:$NUM_TRAIN_NODES}")

# Choose the first training node as the rendezvous head node
HEAD_NODE=${TRAIN_NODES[0]}
MASTER_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)

MASTER_PORT=6000
echo "Head Node IP: $MASTER_IP, port: ${MASTER_PORT}"

# Create a comma-separated list of training nodes for srun
TRAIN_NODES_LIST=$(IFS=, ; echo "${TRAIN_NODES[*]}")

export NCCL_DEBUG=INFO
echo "environment: $(env | grep NCCL)"



export DEBUG_MODE=True
export WANDB_PROJECT=RL4SGG

export DATA_PATH="JosephZ/vg150_train_sgg_prompt"
export MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"


# Training setup
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

MAX_PIXELS=$((512 * 28 * 28))

echo "Start training script..."

LAUNCHER="accelerate launch \
    --multi_gpu \
    --num_machines $NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip "$MASTER_IP" \
    --main_process_port $MASTER_PORT \
    --num_processes $WORLD_SIZE \
    --machine_rank \$SLURM_PROCID \
    --role $SLURMD_NODENAME: \
    --rdzv_conf rdzv_backend=c10d \
    --max_restarts 0 \
    --tee 3 \
    --config_file local_scripts/fsdp.yaml \
"

CMD=" \
    open_r1/grpo.py \
    --output_dir models/qwen2vl-fsdp-g8 \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name $DATA_PATH \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
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
    --run_name Qwen2VL-7B-GRPO-fsdp-G8 \
    --save_steps 100 \
    --num_generations 8
"


srun --jobid $SLURM_JOB_ID bash -c "$LAUNCHER $CMD"

echo "END TIME: $(date)"
