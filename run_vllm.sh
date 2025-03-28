#!/bin/bash

#SBATCH --job-name=VLLM
#SBATCH --time=24:00:00

#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=rtx_4090:4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=25000M
#SBATCH --output=DP-VLLM_%j_%N.out

# ------------------ Config ------------------
MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
TP_SIZE=4                 # Tensor parallelism per node (same as GPUs per node)
PORT_BASE=8888

# ------------------ Environment ------------------
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
DP_WORLD_SIZE=${#nodes[@]}      # Data Parallel world size

head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
MASTER_PORT=$(shuf -i 20000-40000 -n 1)
MASTER_IP=${head_node_ip}

echo "Head node IP: ${MASTER_IP}, Port: ${MASTER_PORT}"
echo "DP_WORLD_SIZE : ${DP_WORLD_SIZE}"
echo "Node list: ${nodes[@]}"

echo "All node IPs:"
> ip_list.txt  # Truncate the file to start fresh

for node in "${nodes[@]}"; do
    ip=$(srun --nodes=1 --ntasks=1 -w "$node" hostname --ip-address)
    echo "$ip" >> ip_list.txt
done


# ------------------ Launch per-node ------------------
for (( i=0; i<${DP_WORLD_SIZE}; i++ ))
do
    node=${nodes[$i]}
    echo "Launching on node $node with rank $i"

    srun --nodes=1 --nodelist=${node} \
    bash -c "
        export RANK=$i
        export DP_WORLD_SIZE=$DP_WORLD_SIZE
        export TP_SIZE=$TP_SIZE
        export MASTER_ADDR=$MASTER_IP
        export MASTER_PORT=$MASTER_PORT

        python vllm_server_v2.py \
            --model '${MODEL_PATH}' \
            --gpu_memory_utilization 0.9 \
            --dtype 'bfloat16' \
            --max_model_len 4096 \
            --tensor_parallel_size ${TP_SIZE} \
            --host '0.0.0.0' \
            --port ${PORT_BASE}
    " &
done

wait
