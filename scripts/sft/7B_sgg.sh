#!/bin/bash



#SBATCH --job-name=SFT_7B_psg
#SBATCH --time=12:00:00

# 4x A100

#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288

#SBATCH --account=a-a03
#SBATCH --mail-user="zychen.uestc@gmail.com"
#SBATCH --mail-type=ALL
#SBATCH --output=SFT-7B_%j_%N.out


set -x
MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
echo "MASTER_ADDR:$MASTER_ADDR"


# Set NODE_RANK from SLURM environment variable
export NODE_RANK=${SLURM_NODEID}

export GPUS_PER_NODE=4

export WANDB_PROJECT=RL4SGG

DATASET=JosephZ/psg_train_sg

RUN_NAME="qwen2vl-7b-sft-open-psg-bs128-e3-merged-gh200"
export OUTPUT_DIR="${SCRATCH}/models/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

# batch size=4 * 2 * 16 = 128
srun torchrun --nnodes ${SLURM_NNODES} \
    --nproc_per_node $GPUS_PER_NODE \
    --node_rank $NODE_RANK \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR}:29500 \
    src/sft_sgg.py \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --dataset_name $DATASET \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.05 \
    --max_grad_norm 0.3 \
    --logging_steps 1 \
    --bf16 true\
    --tf32 true\
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 3 \
    --run_name  $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --torch_dtype bfloat16 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config local_scripts/fsdp_config.json \
    --output_dir $OUTPUT_DIR \
    --seed 42

