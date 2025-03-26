#!/bin/bash



export DATA_PATH="JosephZ/vg150_train_sgg_prompt"

export CUDA_VISIBLE_DEVICES=6

accelerate launch --num_processes=1 open_r1/grpo.py \
    --output_dir models/qwen2vl-sgg-g8 \
    --model_name_or_path "Qwen/Qwen2-VL-2B-Instruct" \
    --dataset_name $DATA_PATH \
    --deepspeed ./local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --max_completion_length 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --use_vllm false \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --max_pixels 401408 \
    --temperature 0.7 \
    --top_p 0.01 \
    --top_k 10 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-SGG-G8 \
    --save_steps 100 \
    --num_generations 2
