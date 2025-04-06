#!/bin/bash


export CUDA_VISIBLE_DEVICES=4,5,6,7

export NODE_RANK=0 
export GPUS_PER_NODE=4

export WANDB_PROJECT=RL4SGG


# 16*8
torchrun --nnodes 1 \
    --nproc_per_node $GPUS_PER_NODE \
    --node_rank $NODE_RANK \
    src/sft_sgg.py \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --dataset_name JosephZ/vg150_train_sgg_prompt \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.05 \
    --max_grad_norm 0.3 \
    --logging_steps 1 \
    --bf16 true\
    --tf32 true\
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 3 \
    --run_name Qwen2-VL-7B_vg150_sgg_b128_normal_close_e3 \
    --save_steps 500 \
    --save_only_model true \
    --torch_dtype bfloat16 \
    --output_dir models/qwen2vl-7b-sft-vg150-b128-normal-close-e3\
    --fsdp "full_shard auto_wrap" \
    --fsdp_config local_scripts/fsdp_config.json \
    --seed 42 \
    --use_liger True 


