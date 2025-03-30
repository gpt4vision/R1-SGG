#!/bin/bash

#SBATCH --job-name=VLLM
#SBATCH --time=24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=rtx_4090:8
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=25000M
#SBATCH --output=VLLM_%j_%N.out

# ------------------ Config ------------------
MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
TP_SIZE=4                   # Tensor parallelism (4 GPUs per process)
PORT_BASE=8000              # Base port to offset by local rank

# ------------------ Environment ------------------
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
DP_WORLD_SIZE=$(( ${#nodes[@]} * 2 ))   # 2 processes per node

head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
MASTER_PORT=$(shuf -i 20000-40000 -n 1)
MASTER_IP=${head_node_ip}

echo "Head node IP: ${MASTER_IP}, Port: ${MASTER_PORT}"
echo "DP_WORLD_SIZE : ${DP_WORLD_SIZE}"
echo "Node list: ${nodes[@]}"

# ------------------ Export IPs and Ports ------------------
IP_FILE=ip_port_test.txt

> ${IP_FILE}  # Reset output list

RANK=0
for node in "${nodes[@]}"; do
    node_ip=$(srun --nodes=1 --ntasks=1 -w "$node" hostname --ip-address)

    for local_rank in 0 1; do
        PORT=$((PORT_BASE + local_rank))
        echo "${node_ip}:${PORT}" >> ${IP_FILE}
    done
done

# ------------------ Launch per-process ------------------
RANK=0
for node in "${nodes[@]}"; do
    echo "Launching 2 ranks on node $node"

    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 -w "$node" \
    bash -c "
        for local_rank in 0 1; do
            (
                export RANK=\$(( ${RANK} + local_rank ))
                export DP_WORLD_SIZE=${DP_WORLD_SIZE}
                export TP_SIZE=${TP_SIZE}
                export MASTER_ADDR=${MASTER_IP}
                export MASTER_PORT=${MASTER_PORT}

                export CUDA_VISIBLE_DEVICES=\$(seq -s, \$(( local_rank * 4 )) \$(( local_rank * 4 + 3 )))
                PORT=\$(( ${PORT_BASE} + local_rank ))

                echo \"Starting rank \$RANK on $node with CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES, port \$PORT\"

                python vllm_server_v2.py \
                    --model '${MODEL_PATH}' \
                    --gpu_memory_utilization 0.85 \
                    --dtype 'bfloat16' \
                    --max_model_len 4096 \
                    --tensor_parallel_size ${TP_SIZE} \
                    --host '0.0.0.0' \
                    --port \$PORT
            ) &
        done
        wait
    " &

    RANK=$((RANK + 2))
done

wait
