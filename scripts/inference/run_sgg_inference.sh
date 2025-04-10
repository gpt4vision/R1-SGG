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
#SBATCH --mail-user="zychen.uestc@gmail.com" 
#SBATCH --mail-type=ALL

# Get node list and determine head node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head Node IP: $head_node_ip"

# Set NODE_RANK from SLURM environment variable
export NODE_RANK=${SLURM_NODEID}
export GPUS_PER_NODE=8






MODEL_NAME=$1
OUTPUT_DIR=$2
USE_CATS=$3     # true/false
PROMPT_TYPE=$4  # true/false

BATCH_SIZE=1

echo "MODEL_NAME: $MODEL_NAME, OUTPUT_DIR: $OUTPUT_DIR"
echo "USE_CATS: $USE_CATS, PROMPT_TYPE: $PROMPT_TYPE"

ARGS="--dataset JosephZ/vg150_val_sgg_prompt --model $MODEL_NAME --output_dir $OUTPUT_DIR --max_model_len 4096 --batch_size $BATCH_SIZE"


if [ "$PROMPT_TYPE" == "true" ]; then
  ARGS="$ARGS --use_think_system_prompt"
fi

if [ "$USE_CATS" == "true" ]; then
  ARGS="$ARGS --use_predefined_cats"
fi

echo "ARGS:$ARGS"

srun torchrun --nnodes ${SLURM_NNODES} \
  --nproc_per_node $GPUS_PER_NODE \
  --node_rank $NODE_RANK \
  --rdzv_id $RANDOM \
  --rdzv_backend c10d \
  --rdzv_endpoint ${head_node_ip}:29500 \
  src/sgg_inference_vllm.py -- $ARGS
