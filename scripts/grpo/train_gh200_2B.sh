#!/bin/bash


#SBATCH --job-name=2B_bs32_gh200
#SBATCH --time=12:00:00

#SBATCH --nodes=2  # 2 nodes, each has 4x GH200                   
#SBATCH --ntasks=2                   # Total tasks equals total nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288 # fixed for GH200

#SBATCH --output=RL_gh200_%j_%N.out


# ---------- Environment Setup ----------
export NCCL_ASYNC_ERROR_HANDLING=1
export DEBUG_MODE=True
export WANDB_PROJECT=RL4SGG


GPUS_PER_NODE=4
GROUP_SIZE=8
MODEL_PATH="Qwen/Qwen2-VL-2B-Instruct"
DATA_PATH="JosephZ/vg150_train_sgg_prompt"
RUN_NAME="qwen2vl-2b-grpo-g${GROUP_SIZE}-n1-temp1-topk50-bs32-gh200"
export OUTPUT_DIR="${SCRATCH}/models/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

MAX_PIXELS=$((512 * 28 * 28))

export LOG_PATH=${OUTPUT_DIR}/debug.log

MASTER_PORT=29500

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NUM_TRAIN_NODES=${#NODELIST[@]}
TRAIN_NODES_LIST=("${NODELIST[@]:0:$NUM_TRAIN_NODES}")

# Choose the first training node as the rendezvous head node
HEAD_NODE=${TRAIN_NODES_LIST[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
echo "Head Node IP: $HEAD_NODE_IP"



#  batch size: PER_DEVICE(16) * ACC(2) *  GPU (4) * NODE(2) // GROUP_SIZE(8) = 32
TRAIN_CMD="open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATA_PATH} \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --custom_per_device_train_batch_size 16 \
    --deepspeed ./local_scripts/zero2_offload.json \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-7 \
    --logging_steps 1 \
    --use_vllm true \
    --use_local_vllm true\
    --bf16 true\
    --tf32 true\
    --report_to wandb \
    --gradient_checkpointing true \
    --max_pixels ${MAX_PIXELS} \
    --temperature 1 \
    --top_p 0.9 \
    --top_k 50 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --num_generations ${GROUP_SIZE} \
    --num_iterations 1 \
    --beta 0.0\
    --vllm_max_model_len 4096 \
    --vllm_gpu_memory_utilization 0.2 \
    --save_only_model false \
    --seed 42"

    
echo "start training..."

srun --nodes=${NUM_TRAIN_NODES} --nodelist="${TRAIN_NODES_LIST}" \
    torchrun --nnodes ${NUM_TRAIN_NODES} --nproc_per_node ${GPUS_PER_NODE} \
    --node_rank ${SLURM_NODEID} \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint ${HEAD_NODE_IP}:${MASTER_PORT} \
    ${TRAIN_CMD}
