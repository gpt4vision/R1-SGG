#!/bin/bash



export DATA_PATH="JosephZ/vg150_train_sgg_prompt"

export CUDA_VISIBLE_DEVICES=0

accelerate launch --num_processes=1 open_r1/grpo.py \
    --output_dir models/qwen2vl-sgg-g8 \
    --model_name_or_path "Qwen/Qwen2-VL-2B-Instruct" \
    --dataset_name $DATA_PATH \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --use_vllm true \
    --use_local_vllm true\
    --use_liger_loss true\
    --vllm_gpu_memory_utilization 0.25\
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --max_pixels 401408 \
    --temperature 0.7 \
    --top_p 0.01 \
    --top_k 1 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-SGG-debug \
    --save_steps 100 \
    --num_generations 2
