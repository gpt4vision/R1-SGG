#!/bin/bash



#SBATCH --job-name=SFT_2B_psg
#SBATCH --time=12:00:00

# 4x A100

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288

#SBATCH --mail-user="zychen.uestc@gmail.com"
#SBATCH --mail-type=ALL
#SBATCH --output=SFT-2B_%j_%N.out

# Get node list and determine head node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)


#DATASET=JosephZ/vg150_train_sgg_prompt
DATASET=JosephZ/psg_train_sg


echo "Head Node IP: $head_node_ip"

# Set NODE_RANK from SLURM environment variable
export NODE_RANK=${SLURM_NODEID}

export GPUS_PER_NODE=4

export WANDB_PROJECT=RL4SGG

RUN_NAME="qwen2vl-2b-sft-open-psg-bs128-e6-merged-gh200"
export OUTPUT_DIR="${SCRATCH}/models/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"


# batch size=4 * 2 * 16 = 128
srun torchrun --nnodes ${SLURM_NNODES} \
    --nproc_per_node $GPUS_PER_NODE \
    --node_rank $NODE_RANK \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint ${head_node_ip}:29500 \
    src/sft_sgg.py \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name $DATASET \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4\
    --gradient_accumulation_steps 1\
    --warmup_ratio 0.05 \
    --max_grad_norm 0.3 \
    --logging_steps 1 \
    --bf16 true\
    --tf32 true\
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 6 \
    --run_name  $RUN_NAME \
    --save_steps 500 \
    --save_only_model true \
    --torch_dtype bfloat16 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config local_scripts/fsdp_config.json \
    --output_dir $OUTPUT_DIR \
    --seed 42
