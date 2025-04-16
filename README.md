# R1-SGG: Compile Scene Graphs with Reinforcement Learning

## Setup Environment
```
bash install.sh
```
the main dependencies are 
```
- torch == 2.5.1 or 2.5.0,  (cu124, optional)
- transformers: supports qwen2vl, qwen2.5vl
- trl
- vLLM
```

## Dataset
```
from datasets import load_dataset

db_train = load_dataset("JosephZ/vg150_train_sgg_prompt")["train"]
db_val = load_dataset("JosephZ/vg150_val_sgg_prompt")["train"]
```
we transformed VG150 into datasets format with keys: "image_id", "image", "prompt_open", "prompt_close", "objects", and "relationships".

## Supported Models
- [x] Qwen/Qwen2-VL-2B-Instruct 
- [x] Qwen/Qwen2-VL-7B-Instruct
- [ ] Qwen/Qwen2.5-VL-3B-Instruct 
- [ ] Qwen/Qwen2.5-VL-7B-Instruct 


## Training with SFT
For slurm users,
```
sbatch scripts/sft/7B_sgg.sh 
```
For local machine,
```
bash scripts/sft_loca/7B_sgg.sh
```



## Training with GRPO
For A100 GPUs, to train a 2B model via
```
sbatch scripts/grpo/train_a100_2B.sh
```
which requires about 12 hours with 16x A100 GPUs.


For GH200 GPUs,
```
sbatch scripts/grpo/train_gh200.sh
```
which requires about 16 hours with 16x GH200 GPUs.

with these large-memory GPUs (> 80GB), we allocate one vLLM server at each training process to reduce the communication latency and speedup the sampling process.


If you have lots of GPUs like RTX_3090/RTX_4090, you can use 
```
sbatch scripts/grpo/train_fused.sh
```
with Zero3, you can train 7B model on 24GB GPUs but the training speed is slow as the communication is the bottleneck (I have tried to use 120 RTX_4090 GPUs. It is crazy, but the communication latency is significant due to RTX_4090 does not have NCCL support.)

## Inference
- To test models trained with SFT, 
```
bash scripts/inference/run_sgg_inference.sh $MODEL_NAME $OUTPUT_DIR
```
If the model trained with predefined categories (i.e., with "--use_predefined_cats"), add the third parameter to the script
```
bash scripts/inference/run_sgg_inference.sh $MODEL_NAME $OUTPUT_DIR true
```

- To test models trained with GRPO,
```
bash scripts/inference/run_sgg_inference.sh $MODEL_NAME $OUTPUT_DIR false/true  true
```

then, run the evaluation via
```
python src/sgg_gather_preds.py $OUTPUT_DIR sgg_pred_results.json
python src/vg150_eval.py sgg_pred_results.json
```





## Acknowledgement
The GRPOTrainer used in this project is based on trl's [GRPOTrainer](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py),
and we extend it to support multimodal inputs.

## Citation


