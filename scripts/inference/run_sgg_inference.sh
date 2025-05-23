#!/bin/bash



export GPUS_PER_NODE=4


DATASET=$1
MODEL_NAME=$2
OUTPUT_DIR=$3
USE_CATS=$4     # true/false
PROMPT_TYPE=$5  # true/false

BATCH_SIZE=${6:-8}

echo "MODEL_NAME: $MODEL_NAME, OUTPUT_DIR: $OUTPUT_DIR"
echo "USE_CATS: $USE_CATS, PROMPT_TYPE: $PROMPT_TYPE"

ARGS="--dataset $DATASET --model $MODEL_NAME --output_dir $OUTPUT_DIR --max_model_len 4096 --batch_size $BATCH_SIZE"


if [ "$PROMPT_TYPE" == "true" ]; then
  ARGS="$ARGS --use_think_system_prompt"
fi

if [ "$USE_CATS" == "true" ]; then
  ARGS="$ARGS --use_predefined_cats"
fi

echo "ARGS:$ARGS"

torchrun --nnodes 1 \
  --nproc_per_node $GPUS_PER_NODE \
  --node_rank 0 \
  src/sgg_inference_vllm.py -- $ARGS
