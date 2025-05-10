#!/bin/bash

export HF_HOME=$SCRATCH/huggingface
# ---------- Environment Setup ----------
export NCCL_ASYNC_ERROR_HANDLING=1
export DEBUG_MODE=True
export WANDB_PROJECT=RL4SGG


MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
DATA_PATH="JosephZ/vg150_train_sgg_prompt"

TP_SIZE=1
PORT_BASE=8000

MAX_PIXELS=$((512 * 28 * 28))




HEAD_NODE_IP=0.0.0.0
MASTER_PORT=29500


   

server_ip=$(hostname -I | awk '{print $1}')


# Launch vLLM servers
for i in {0..1}; do
    log_file="vllm_server_${i}.log"
    port=$((PORT_BASE + i))
    CUDA_VISIBLE_DEVICES=${i} python src/vllm_server_v2.py \
        --model "${MODEL_PATH}" \
        --gpu_memory_utilization 0.9 \
        --enable_prefix_caching true \
        --dtype 'bfloat16' \
        --max_model_len 4096 \
        --tensor_parallel_size "${TP_SIZE}" \
        --host '0.0.0.0' \
        --port "${port}" > "${log_file}" 2>&1 &
done

echo "Waiting for vLLM servers to initialize..."
#sleep 60

# Run tests
for i in {2..3}; do
    log_file="vllm_client_${i}.log"
    port=$((PORT_BASE + i - 2))
    group_port=$(( 51200 + i))
    CUDA_VISIBLE_DEVICES=${i} python tests/test_vllm.py \
        --hosts ${server_ip} \
        --server_port "${port}" \
        --group_port ${group_port}\
        --model_name_or_path "${MODEL_PATH}" > "${log_file}" 2>&1 &
done

  
