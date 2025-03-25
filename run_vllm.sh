#!/bin/bash

#SBATCH --job-name=VLLM
#SBATCH --time=24:00:00
#SBATCH --nodes=1                    # 4 training nodes + 1 vLLM node = 5 nodes
#SBATCH --ntasks=1                   # Total tasks equals total nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=rtx_4090:4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=25000M
#SBATCH --output=VLLM_%j_%N.out

# Define node counts
NUM_VLLM_NODE=1
GPUS_PER_NODE=4

# Get the list of allocated nodes
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# Assign vLLM server node (the last node in the allocation)
VLLM_NODE_A=${NODELIST[0]}



# Define HOST and PORT for the vLLM server
HOST_A=$VLLM_NODE_A
PORT_A=8888

HOST_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HOST_A" hostname --ip-address)

echo "HOST_A:$HOST_A, PORT_A:$PORT_A, IP:$HOST_NODE_IP"

# Start vLLM server on the vLLM node using 4 GPUs (adjust as needed)
srun --nodes=1 --nodelist="${VLLM_NODE_A}" \
    python vllm_server.py \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --gpu_memory_utilization 0.9 \
    --dtype "bfloat16" \
    --max_model_len 8192 \
    --tensor_parallel_size 4 \
    --pipeline_parallel_size 1 \
    --host 0.0.0.0 \
    --port $PORT_A 

