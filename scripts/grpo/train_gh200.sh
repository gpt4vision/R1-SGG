#!/bin/bash


#SBATCH --job-name=7B_GH200_zero_lr2x_fp8
#SBATCH --time=12:00:00

#SBATCH --exclude=nid006792,nid007085

#SBATCH --nodes=8  # 4 nodes, each has 4x GH200                   
#SBATCH --ntasks=8                   # Total tasks equals total nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288 # fixed for GH200


#SBATCH --account=a-a03
#SBATCH --partition=normal
#SBATCH --output=RL_gh200_%j_%N.out
#SBATCH --mail-user="zychen.uestc@gmail.com" --mail-type=ALL


set -x
# ---------- Environment Setup ----------
export NCCL_ASYNC_ERROR_HANDLING=1
export DEBUG_MODE=True
export WANDB_PROJECT=RL4SGG


GPUS_PER_NODE=4
GROUP_SIZE=8
MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
DATA_PATH="JosephZ/vg150_train_sgg_prompt"
RUN_NAME="qwen2vl-7b-grpo-2k-lr6e-7-g8-n1-bs32-fp8-gh200"
export OUTPUT_DIR="${SCRATCH}/models/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

export LOG_PATH=${OUTPUT_DIR}/debug.log

export STRICT_FORMAT=True

MAX_PIXELS=$((512 * 28 * 28))


MASTER_PORT=29500

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NUM_TRAIN_NODES=${#NODELIST[@]}
TRAIN_NODES_LIST=("${NODELIST[@]:0:$NUM_TRAIN_NODES}")

# Choose the first training node as the rendezvous head node
##HEAD_NODE=${TRAIN_NODES_LIST[0]}
##HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
##echo "Head Node IP: $HEAD_NODE_IP"

MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
echo "MASTER_ADDR: $MASTER_ADDR"



# GH200 has a very high bandwidth between CPU and GPU, we should use it!
# zero2:
# bsz_per_devie=16, OOM; Ok,  with CPU offload for optimizer, ~60h with 3x GPUs
# bsz_per_devie=8, 386s for 30 steps, ~60h with 3x GPUs
# bsz_per_devie=16, ~40h with 4x GPUs
#
#  batch size: 16*1*4*4 //8=32
TRAIN_CMD="open_r1/grpo.py \
    --task_type sgg \
    --use_fp8 true \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATA_PATH} \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --custom_per_device_train_batch_size 8 \
    --deepspeed ./local_scripts/zero2_offload.json \
    --gradient_accumulation_steps 1 \
    --learning_rate 6e-7 \
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
    --beta 0.0 \
    --vllm_max_model_len 4096 \
    --vllm_gpu_memory_utilization 0.2 \
    --ddp_timeout 3600 \
    --save_only_model false"

    
echo "start training with CMD=${TRAIN_CMD} ..."

WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_TRAIN_NODES))


LAUNCHER="accelerate launch \
    --multi_gpu \
    --num_machines $NUM_TRAIN_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port $MASTER_PORT \
    --num_processes $WORLD_SIZE \
    --machine_rank $SLURM_PROCID \
    --role $SLURMD_NODENAME: \
    --rdzv_conf rdzv_backend=c10d \
    --rdzv_timeout 3600 \
    --max_restarts 0 \
    --tee 3 \
    --mixed_precision fp8 \
"


srun --jobid $SLURM_JOB_ID bash -c "$LAUNCHER $TRAIN_CMD"

#srun torchrun --nnodes ${NUM_TRAIN_NODES} --nproc_per_node ${GPUS_PER_NODE} \
#    --node_rank ${SLURM_NODEID} \
#    --rdzv_id $RANDOM \
#    --rdzv_backend c10d \
#    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
#    --rdzv_timeout 3600 \
#    ${TRAIN_CMD}
