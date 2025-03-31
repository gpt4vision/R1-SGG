#!/bin/bash



#SBATCH --job-name=7b-eval
#SBATCH --time=24:00:00

## Using 2 nodes (each node has 8x4090 GPUs)

#GPU types:
# - a100-pcie-40gb
# - a100_80gb
# - rtx_4090
# - rtx_3090

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=rtx_4090:8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=25000M
#SBATCH --output=%j_%N.out
#SBATCH --mail-user="zychen.hk@icloud.com" 
#SBATCH --mail-type=ALL

# Get node list and determine head node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head Node IP: $head_node_ip"

# Set NODE_RANK from SLURM environment variable
export NODE_RANK=${SLURM_NODEID}

export GPUS_PER_NODE=8




# Start with port 30000 and search for an available port.
port=29500
while lsof -i :$port >/dev/null 2>&1; do
    port=$((port+10))
done

echo "using port $port"


#MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
#MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
#MODEL_NAME="models/qwen2vl-7b-sft-vg150-sgg/checkpoint-2000/"
#MODEL_NAME="models/qwen2vl-7b-sft-vg150-sgg-cot/checkpoint-180/"
#MODEL_NAME="models/qwen2vl-7b-sft-vg150-sgg/checkpoint-100/"
#MODEL_NAME="models/qwen2vl-7b-sft-vg150-sgg-nopack/checkpoint-100/"
#MODEL_NAME="models/qwen2vl-7b-sft-vg150-sgg-nopack/checkpoint-1757/"
#MODEL_NAME="models/qwen2vl-7b-sft-vg150-sgg-nopack/checkpoint-1700/"
#MODEL_NAME="models/qwen2vl-7b-sft-vg150-sgg/checkpoint-1700/"
#MODEL_NAME="models/qwen2vl-7b-sft-vg150-sgg-close/checkpoint-700/"
#MODEL_NAME="models/qwen2vl-7b-sft-vg150-b64-e3/checkpoint-1900/"
#MODEL_NAME="models/qwen2vl-7b-sft-vg150-b64-close-e3/checkpoint-1500/"

MODEL_NAME=$1

echo "MODEL_NAME: $MODEL_NAME"


#args: Namespace(dataset='JosephZ/vg150_val_sgg_prompt', model_name='Qwen/Qwen2.5-VL-7B-Instruct', output_dir='logs/sgg-infer/7b-baseline-2.5-close')


srun torchrun --nnodes ${SLURM_NNODES} \
    --nproc_per_node $GPUS_PER_NODE \
    --node_rank $NODE_RANK \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint ${head_node_ip}:29500 \
    src/sgg_inference_vllm.py -- \
    --dataset "JosephZ/vg150_val_sgg_prompt" \
    --model_name $MODEL_NAME \
    --output_dir "logs/sgg-vllm/7b-zero-rl-500"
