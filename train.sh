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
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

# wait for vLLM servers
#sleep 60

# Read IPs from file and join them with commas
ip_str=$(paste -sd, ip_list.txt)

# Print or use the resulting string
echo "vLLM servers: $ip_str"

# Define node counts
NUM_TRAIN_NODES=16
GPUS_PER_NODE=8

# Get the list of allocated nodes
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# Assign training nodes (first NUM_TRAIN_NODES nodes)
TRAIN_NODES=("${NODELIST[@]:0:$NUM_TRAIN_NODES}")

# Choose the first training node as the rendezvous head node
HEAD_NODE=${TRAIN_NODES[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
echo "Head Node IP: $HEAD_NODE_IP"

# Create a comma-separated list of training nodes for srun
TRAIN_NODES_LIST=$(IFS=, ; echo "${TRAIN_NODES[*]}")

# Define HOST and PORT for the vLLM server
PORT_A=8888


export DEBUG_MODE=True
export WANDB_PROJECT=RL4SGG

export DATA_PATH="JosephZ/vg150_train_sgg_prompt"
export MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"

export NODE_RANK=${SLURM_NODEID}  # Provided by SLURM

# Launch distributed training on the training nodes using 8 GPUs per node
srun --nodes=${NUM_TRAIN_NODES} --nodelist="${TRAIN_NODES_LIST}" \
    torchrun --nnodes ${NUM_TRAIN_NODES} --nproc_per_node ${GPUS_PER_NODE} \
    --node_rank $NODE_RANK \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint ${HEAD_NODE_IP}:29500 \
    open_r1/grpo.py \
    --output_dir models/qwen2vl-zero-g8 \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name $DATA_PATH \
    --deepspeed ./local_scripts/zero3.json \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --use_vllm true \
    --vllm_server_host ${ip_str} \
    --vllm_server_port ${PORT_A} \
    --vllm_server_timeout 360 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --max_pixels 401408 \
    --temperature 0.7 \
    --top_p 0.01 \
    --top_k 1 \
    --num_train_epochs 1 \
    --run_name Qwen2VL-7B-GRPO-zero-G8 \
    --save_steps 100 \
    --num_generations 8
