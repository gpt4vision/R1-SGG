#!/bin/bash





export GPUS_PER_NODE=4

export WANDB_PROJECT=RL4SGG


# batch size=4 * 2 * 16 = 128
torchrun --nnodes 1 \
    --nproc_per_node $GPUS_PER_NODE \
    --node_rank 0 \
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
    --run_name Qwen2-VL-7B_vg150_sgg_b128_open_e3 \
    --save_steps 100 \
    --save_only_model true \
    --torch_dtype bfloat16 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config local_scripts/fsdp_config.json \
    --output_dir models/qwen2vl-7b-sft-vg150-b128-open-e3 \
    --seed 42

