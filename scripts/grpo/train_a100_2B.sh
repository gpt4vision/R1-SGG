#!/bin/bash


#SBATCH --job-name=A100_2B_1k_lr6e-7_psg_debug
#SBATCH --time=00:30:00

#SBATCH --exclude=nid002289,nid002325
#SBATCH --nodes=2  # 4 nodes, each has 4x A100  
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128

#SBATCH --partition=normal
#SBATCH --output=RL_A100_%j_%N.out
#SBATCH --mail-user="zychen.uestc@gmail.com" --mail-type=ALL


set -x
# ---------- Environment Setup ----------
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export DEBUG_MODE=True
export WANDB_PROJECT=RL4SGG

export NCCL_DEBUG=INFO


GPUS_PER_NODE=4
GROUP_SIZE=8
MODEL_PATH="Qwen/Qwen2-VL-2B-Instruct"
#DATA_PATH="JosephZ/vg150_train_sgg_prompt"
DATA_PATH="JosephZ/psg_train_sg"

RUN_NAME="qwen2vl-2b-grpo-g8-n1-bs32-1k-lr6e-7-psg-debug-A100-SXM4"
export OUTPUT_DIR="${SCRATCH}/models/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

export LOG_PATH=${OUTPUT_DIR}/debug.log

export FORMAT_REWARD_WEIGHT=1.0
export STRICT_FORMAT=True

MAX_PIXELS=$((512 * 28 * 28))


MASTER_PORT=29500

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NUM_TRAIN_NODES=${#NODELIST[@]}
TRAIN_NODES_LIST=("${NODELIST[@]:0:$NUM_TRAIN_NODES}")

# Choose the first training node as the rendezvous head node
HEAD_NODE=${TRAIN_NODES_LIST[0]}

#MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
#MASTER_ADDR=$(echo "${SLURM_NODELIST}" | sed 's/[],].*//g; s/\[//g')

MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
echo "MASTER_ADDR: $MASTER_ADDR"



# batch size: PER_GPU(4)*GPUS(4)*NODES(4)*ACC(4) // GROUP_SIZE(8) = 32
# local vLLM: 80G*0.2=16G
#
# ['format_reward', 'node_acc_reward', "node_box_reward",  "edge_reward"]
TRAIN_CMD="open_r1/grpo.py \
    --task_type sgg \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATA_PATH} \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --custom_per_device_train_batch_size 4 \
    --deepspeed ./local_scripts/zero2.json \
    --gradient_accumulation_steps 4 \
    --learning_rate 6e-7 \
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
    --num_train_epochs 1.0 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --num_generations ${GROUP_SIZE} \
    --num_iterations 1 \
    --beta 0.0 \
    --vllm_max_model_len 4096 \
    --vllm_gpu_memory_utilization 0.2 \
    --save_only_model true\
    --seed 42"

    
echo "start training..."

srun --nodes=${NUM_TRAIN_NODES} --nodelist="${TRAIN_NODES_LIST}" \
    torchrun --nnodes ${NUM_TRAIN_NODES} --nproc_per_node ${GPUS_PER_NODE} \
    --node_rank ${SLURM_NODEID} \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    ${TRAIN_CMD}
