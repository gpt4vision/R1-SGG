#!/bin/bash


#SBATCH --job-name=GRPO_train_A100
#SBATCH --time=24:00:00

#SBATCH --nodes=4  # 4 nodes, each has 4x A100  
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128

#SBATCH --account=a-a03
#SBATCH --partition=normal
#SBATCH --output=RL_A100_%j_%N.out
#SBATCH --mail-user="zychen.uestc@gmail.com" --mail-type=ALL


# ---------- Environment Setup ----------
export NCCL_ASYNC_ERROR_HANDLING=1
export DEBUG_MODE=True
export WANDB_PROJECT=RL4SGG


GPUS_PER_NODE=4
GROUP_SIZE=8
MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
DATA_PATH="JosephZ/vg150_train_sgg_prompt"
RUN_NAME="qwen2vl-7b-grpo-g${GROUP_SIZE}-n1-temp1-topk50-top0.9"
export OUTPUT_DIR="${SCRATCH}/models/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

MAX_PIXELS=$((512 * 28 * 28))


MASTER_PORT=29500

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NUM_TRAIN_NODES=${#NODELIST[@]}
TRAIN_NODES_LIST=("${NODELIST[@]:0:$NUM_TRAIN_NODES}")

# Choose the first training node as the rendezvous head node
HEAD_NODE=${TRAIN_NODES_LIST[0]}

#MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)

MASTER_ADDR=$(echo "${SLURM_NODELIST}" | sed 's/[],].*//g; s/\[//g')
echo "MASTER_ADDR: $MASTER_ADDR"



# batch size: 4*GPUS(4)*NODES(2)*ACC(8) //8=32
# local vLLM: 80G*0.25=20G
#
TRAIN_CMD="open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATA_PATH} \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 2 \
    --deepspeed ./local_scripts/zero2.json \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --use_vllm true \
    --use_local_vllm true\
    --bf16 true\
    --tf32 true\
    --report_to wandb \
    --gradient_checkpointing true \
    --max_pixels ${MAX_PIXELS} \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --num_generations ${GROUP_SIZE} \
    --num_iterations 1 \
    --beta 0.0\
    --vllm_max_model_len 4096 \
    --vllm_gpu_memory_utilization 0.25\
    --save_only_model false"

    
echo "start training..."

srun --nodes=${NUM_TRAIN_NODES} --nodelist="${TRAIN_NODES_LIST}" \
    torchrun --nnodes ${NUM_TRAIN_NODES} --nproc_per_node ${GPUS_PER_NODE} \
    --node_rank ${SLURM_NODEID} \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    ${TRAIN_CMD}
