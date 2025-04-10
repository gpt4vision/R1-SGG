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

