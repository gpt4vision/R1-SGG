# R1-SGG: Compile Scene Graphs with Reinforcement Learning

**Structured Visual Reasoning with Multimodal LLMs and Reinforcement Learning**  
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

---

## 🚀 Update
- ✅ Support [PSG](https://github.com/Jingkang50/OpenPSG) dataset (bbox format only, not Panoptic)
- ✅ Updated loss implementation
- ✅ Always use `custom_per_device_train_batch_size` instead of `per_device_train_batch_size` for faster sampling under gradient accumulation
- ⚠️ Current loss implementation might still be affected by gradient accumulation: [trl issue #3021](https://github.com/huggingface/trl/issues/3021)

---

## 🛠️ Setup Environment
```bash
bash install.sh
```
Main dependencies:
```bash
- torch == 2.5.0 or 2.5.1 (cu124, optional)
- transformers (supports Qwen2VL, Qwen2.5VL)
- trl
- vLLM
```

---

## 📚 Dataset
Load preprocessed datasets via:
```python
from datasets import load_dataset

db_train = load_dataset("JosephZ/vg150_train_sgg_prompt")["train"]
db_val = load_dataset("JosephZ/vg150_val_sgg_prompt")["train"]
```
or for PSG:
```python
db_train = load_dataset("JosephZ/psg_train_sg")["train"]  # keys: image_id, image, objects, relationships
db_val = load_dataset("JosephZ/psg_test_sg")["train"]
```
We transformed VG150 into HuggingFace Datasets format with keys:
- `image_id`
- `image`
- `prompt_open`
- `prompt_close`
- `objects`
- `relationships`

---

## 🔥 Supported Models
- [x] Qwen/Qwen2-VL-2B-Instruct
- [x] Qwen/Qwen2-VL-7B-Instruct
- [ ] Qwen/Qwen2.5-VL-3B-Instruct
- [ ] Qwen/Qwen2.5-VL-7B-Instruct

---

## 🏋️‍♂️ Training

### Training with Supervised Fine-Tuning (SFT)

For **SLURM users**:
```bash
sbatch scripts/sft/7B_sgg.sh 
```

For **local machines**:
```bash
bash scripts/sft_local/7B_sgg.sh
```
⏱️ Approximate training time:
- 2B models: ~4 hours (4×A100 SXM4 GPUs)
- 7B models: ~10 hours (4×A100 SXM4 GPUs)

---

### Training with Reinforcement Learning (GRPO)

For **A100 GPUs**:
```bash
sbatch scripts/grpo/train_a100_2B.sh
```
(12 hours on 16×A100 GPUs)

For **GH200 GPUs**:
```bash
sbatch scripts/grpo/train_gh200.sh
```
(16 hours on 16×GH200 GPUs)

For clusters with many RTX_3090/4090 GPUs:
```bash
sbatch scripts/grpo/train_fused.sh
```
- Training 7B models on 24GB cards is possible with Zero3, but slow due to communication bottlenecks.
- (Fun fact: training with 120×RTX_4090 is crazy but severely limited by communication latency.)

💡 **Recommended learning rate**: `6e-7`.

---

## 🧪 Inference and Evaluation

### Inference with SFT-trained models:
```bash
bash scripts/inference/run_sgg_inference.sh $MODEL_NAME $OUTPUT_DIR
```
For models trained **with predefined categories**, add `true`:
```bash
bash scripts/inference/run_sgg_inference.sh $MODEL_NAME $OUTPUT_DIR true
```

### Inference with GRPO-trained models:
```bash
bash scripts/inference/run_sgg_inference.sh $DATASET $MODEL_NAME $OUTPUT_DIR false/true true
```

### Evaluation:
```bash
python src/sgg_gather_preds.py $OUTPUT_DIR sgg_pred_results.json
python src/vg150_eval.py $DATASET sgg_pred_results.json
```

---

## 🤝 Acknowledgement
The `GRPOTrainer` used in this project is based on [trl's GRPOTrainer](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py), extended to support multimodal inputs.

---

## 📖 Citation
If you find this work helpful, please cite:
```bibtex
@article{chen2025compile,
  title={Compile Scene Graphs with Reinforcement Learning},
  author={Chen, Zuyao and Wu, Jinlin and Lei, Zhen and Pollefeys, Marc and Chen, Chang Wen},
  journal={arXiv preprint arXiv:2504.13617},
  year={2025}
}
```

---

# ✨ Happy Compiling!
