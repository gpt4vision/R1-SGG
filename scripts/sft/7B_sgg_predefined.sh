#!/bin/bash



#SBATCH --job-name=SFT_7B_close
#SBATCH --time=24:00:00

# 4x A100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128

#SBATCH --mail-user="zychen.uestc@gmail.com"
#SBATCH --mail-type=ALL
#SBATCH --output=SFT-7B-close_%j_%N.out

# Get node list and determine head node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head Node IP: $head_node_ip"

# Set NODE_RANK from SLURM environment variable
export NODE_RANK=${SLURM_NODEID}

export GPUS_PER_NODE=4

export WANDB_PROJECT=RL4SGG


# batch size=4 * 2 * 16 = 128
srun torchrun --nnodes ${SLURM_NNODES} \
    --nproc_per_node $GPUS_PER_NODE \
    --node_rank $NODE_RANK \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint ${head_node_ip}:29500 \
    src/sft_sgg.py \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --dataset_name JosephZ/vg150_train_sgg_prompt \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --warmup_ratio 0.05 \
    --max_grad_norm 0.3 \
    --logging_steps 1 \
    --bf16 true\
    --tf32 true\
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 3 \
    --run_name Qwen2-VL-7B_vg150_sgg_b128_predefined_e3 \
    --save_steps 100 \
    --save_only_model true \
    --torch_dtype bfloat16 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config local_scripts/fsdp_config.json \
    --use_predefined_cats true \
    --output_dir models/qwen2vl-7b-sft-vg150-b128-predefined-e3 \
    --seed 42

