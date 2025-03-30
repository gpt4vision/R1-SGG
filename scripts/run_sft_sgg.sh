#!/bin/bash



#SBATCH --job-name=SFT_vg150_SGG
#SBATCH --time=24:00:00

## Using 2 nodes (each node has 8x4090 GPUs)

#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=rtx_4090:8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=25000M
#SBATCH --mail-user="zuychen@ethz.ch"
#SBATCH --mail-type=ALL
#SBATCH --output=job_%j_%N.out

# Get node list and determine head node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head Node IP: $head_node_ip"

# Set NODE_RANK from SLURM environment variable
export NODE_RANK=${SLURM_NODEID}

export GPUS_PER_NODE=8

export WANDB_PROJECT=RL4SGG


# 8*8
srun torchrun --nnodes ${SLURM_NNODES} \
    --nproc_per_node $GPUS_PER_NODE \
    --node_rank $NODE_RANK \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint ${head_node_ip}:29500 \
    src/sft_sgg.py \
    --deepspeed configs/zero3.json \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --dataset_name JosephZ/vg150_train_sgg_prompt \
    --learning_rate 1.0e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.05 \
    --max_grad_norm 0.3 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing True \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 3 \
    --run_name Qwen2-VL-7B_vg150_sgg_b64_normal_e3 \
    --save_steps 500 \
    --save_only_model true \
    --torch_dtype bfloat16 \
    --output_dir models/qwen2vl-7b-sft-vg150-b64-normal-e3

