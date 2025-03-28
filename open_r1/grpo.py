# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

from trainer import GRPOTrainerV2, GRPOConfig

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from scipy.optimize import linear_sum_assignment
import spacy
from functools import lru_cache

from trl.data_utils import maybe_apply_chat_template

# Set DEBUG_MODE flag and log path once
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
LOG_PATH = os.getenv("LOG_PATH", "debug.log")

# Load spaCy model (with word vectors)
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    from spacy.cli import download
    download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

SEM_WEIGHT = 0.5
IOU_WEIGHT = 0.5


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def extract_answer_content(text: str) -> str:
    """
    Extracts the content between <answer> and </answer> tags.
    If no closing tag is found, extracts everything after the first <answer>.

    Returns:
        str: The extracted content.
    """
    text = text.replace("```", " ").replace("json", " ").strip()

    # Try to find full <answer>...</answer>
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: everything after the first <answer>
    match = re.search(r"<answer>(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else text



@lru_cache(maxsize=None)
def get_doc(word: str):
    return nlp(word)

def category_semantic_similarity(pred_id: str, gt_id: str) -> float:
    # Extract category names from ids (substring before the dot)
    cat_pred = pred_id.split('.')[0].lower()
    cat_gt = gt_id.split('.')[0].lower()
    return get_doc(cat_pred).similarity(get_doc(cat_gt))


def compute_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return 0.0 if unionArea == 0 else interArea / unionArea




def cost_function(pred, gt, sem_weight=SEM_WEIGHT, iou_weight=IOU_WEIGHT):
    assert len(pred['bbox']) == 4, f"Invalid bbox length: {len(pred['bbox'])}"
    iou = compute_iou(pred['bbox'], gt['bbox'])
    sem_sim = category_semantic_similarity(pred['id'], gt['id'])
    return sem_weight * (1.0 - sem_sim) + iou_weight * (1.0 - iou)


def bi_match(groundtruths, predictions, sem_weight=SEM_WEIGHT, iou_weight=IOU_WEIGHT):
    num_gt = len(groundtruths)
    num_pred = len(predictions)
    pad = max(0, num_gt - num_pred)
    cost_matrix = np.zeros((num_pred + pad, num_gt))

    for i, pred in enumerate(predictions):
        for j, gt in enumerate(groundtruths):
            cost_matrix[i, j] = cost_function(pred, gt, sem_weight, iou_weight)
    if pad > 0:
        cost_matrix[num_pred:, :] = 10000  # High cost for padded rows

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = []
    for r, c in zip(row_ind, col_ind):
        if r >= num_pred:
            continue
        assignments.append({
            'groundtruth': groundtruths[c],
            'prediction': predictions[r],
            'cost': cost_matrix[r, c]
        })
    return assignments


def node_reward(completions, solution, **kwargs):
    """Compute node-level rewards."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol in zip(contents, solution):
        reward = 0.0
        match_objects = []
        try:
            gt_objs = sol['objects']
            # gt_rels is not used here but kept for consistency with edge_reward
            preds = json.loads(extract_answer_content(content))
            pred_objs = preds['objects']

            assignments = bi_match(gt_objs, pred_objs, SEM_WEIGHT, IOU_WEIGHT)
            for assign in assignments:
                gt_id = assign['groundtruth']['id']
                pred_entry = assign['prediction']
                if pred_entry is None or pred_entry.get('id') is None:
                    continue
                pred_id = pred_entry['id']
                match_objects.append(
                    f"Groundtruth {gt_id} -> Prediction {pred_id} with cost {assign['cost']:.3f}"
                )
                reward += (SEM_WEIGHT * category_semantic_similarity(gt_id, pred_id) +
                           IOU_WEIGHT * compute_iou(assign['groundtruth']['bbox'], pred_entry['bbox']))
            reward /= len(gt_objs) if gt_objs else 1
        except Exception:
            reward = 0.0

        rewards.append(reward)
        if DEBUG_MODE:
            with open(LOG_PATH, "a") as f:
                f.write(f"------------- {current_time} Node-level Reward {reward:.3f} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"solution: {sol}\n")
                if match_objects:
                    f.write(f"Match objects: {match_objects}\n")
    return rewards


def edge_reward(completions, solution, **kwargs):
    """Compute edge-level rewards."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol in zip(contents, solution):
        reward = 0.0
        match_objects = []
        match_triplets = []
        try:
            gt_objs = sol['objects']
            gt_rels = sol['relationships']
            preds = json.loads(extract_answer_content(content))
            pred_objs = preds['objects']
            pred_rels = preds['relationships']

            assignments = bi_match(gt_objs, pred_objs, SEM_WEIGHT, IOU_WEIGHT)
            map_obj = {}
            for assign in assignments:
                gt_id = assign['groundtruth']['id']
                pred_entry = assign['prediction']
                if pred_entry is None or pred_entry.get('id') is None:
                    continue
                pred_id = pred_entry['id']
                match_objects.append(
                    f"Groundtruth {gt_id} -> Prediction {pred_id} with cost {assign['cost']:.3f}"
                )
                map_obj[gt_id] = pred_id

            pred_triplets = { (rel['subject'], rel['object']): rel['predicate'] for rel in pred_rels }
            for gt_rel in gt_rels:
                sub, obj = gt_rel['subject'], gt_rel['object']
                if sub not in map_obj or obj not in map_obj:
                    continue
                sub_mapped = map_obj[sub]
                obj_mapped = map_obj[obj]
                if (sub_mapped, obj_mapped) in pred_triplets:
                    pred_pred = pred_triplets[(sub_mapped, obj_mapped)]
                    reward += category_semantic_similarity(gt_rel['predicate'], pred_pred)
                    match_triplets.append(
                        f"GT triplet: {sub_mapped} -> {gt_rel['predicate']} -> {obj_mapped}, "
                        f"Pred: {sub_mapped} -> {pred_pred} -> {obj_mapped}"
                    )
            reward /= len(gt_rels) if gt_rels else 1
        except Exception:
            reward = 0.0

        rewards.append(reward)
        if DEBUG_MODE:
            with open(LOG_PATH, "a") as f:
                f.write(f"------------- {current_time} Edge-level Reward {reward:.3f} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"solution: {sol}\n")
                if match_objects:
                    f.write(f"Match objects: {match_objects}\n")
                if match_triplets:
                    f.write(f"Match triplets: {match_triplets}\n")
    return rewards


def format_reward(completions, **kwargs):
    """
    Reward function that checks if the completion has the correct format:
    - Must contain <think>...</think> and <answer>...</answer>
    - The <answer> content must be a valid JSON dict with keys "objects" and "relationships"
    """
    pattern = r"<think>.*?</think>\s*<answer>(.*?)</answer>"
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for completion in completions:
        content = completion[0]["content"]
        match = re.fullmatch(pattern, content, re.DOTALL)
        reward = 0.0
        if not match:
            rewards.append(0.0)
            continue
        answer_content = match.group(1).strip()
        try:
            answer_json = json.loads(extract_answer_content(answer_content))
            if isinstance(answer_json, dict) and "objects" in answer_json and "relationships" in answer_json:
                reward = 1.0
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except json.JSONDecodeError:
            rewards.append(0.0)

        if DEBUG_MODE:
            with open(LOG_PATH, "a") as f:
                f.write(f"------------- {current_time} Format Reward {reward:.3f} -------------\n")
                f.write(f"content: {content}\n")

    return rewards


# Reward functions registry
reward_funcs_registry = {
    "format": format_reward,
    "node_reward": node_reward,
    "edge_reward": edge_reward,
    # "diversity_reward": diversity_reward
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    script_args.reward_funcs = ['format', 'node_reward', "edge_reward"]
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset = load_dataset(script_args.dataset_name)['train']
    print("len(dataset):", len(dataset))

    def replace_answer_format(item: str) -> str:
        return item.replace("<answer>", "```").replace("</answer>", "```")

    class Collator(object):
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, examples):
            batch = []
            for example in examples:
                prompt = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": replace_answer_format(example["prompt_open"])}
                        ]
                    }
                ]
                gt_objs = example["objects"]
                gt_rels = example["relationships"]
                if not isinstance(gt_objs, (list, tuple)):
                    gt_objs = json.loads(gt_objs)
                if not isinstance(gt_rels, (list, tuple)):
                    gt_rels = json.loads(gt_rels)

                scene_graph = {"objects": gt_objs, "relationships": gt_rels}
                batch.append({"prompt": prompt, "image": example["image"], "solution": scene_graph})

            #prompts_text = [maybe_apply_chat_template(example, self.processor)["prompt"] for example in batch]
            #images = [x["image"] for x in batch]

            #prompt_inputs = self.processor(
            #    text=prompts_text, images=images,
            #    return_tensors="pt", padding=True,
            #    padding_side="left", add_special_tokens=False,
            #)

            #return batch, prompt_inputs            
            return batch

    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    except:
        rank = 0
        world_size = 1

    processor = Qwen2VLProcessor.from_pretrained(model_args.model_name_or_path, 
                    min_pixels=script_args.min_pixels,
                    max_pixels=script_args.max_pixels)
    collator_instance = Collator(processor)

    print("*" * 100)
    print(f"rank={rank}, world size={world_size}, len(dataset)={len(dataset)}, dataset[0]:", collator_instance([dataset[0]]))
    print("use_vllm:", training_args.use_vllm)
    print("data collator:", collator_instance)
    print("*" * 100)


    
    if not hasattr(training_args, "model_init_kwargs") or training_args.model_init_kwargs is None:
        training_args.model_init_kwargs = {
            'torch_dtype': torch.bfloat16,
            'attn_implementation': 'flash_attention_2'
        }

    if not hasattr(training_args, "temperature"):
        training_args.temperature = getattr(script_args, "temperature", 0.9)
    if not hasattr(training_args, "top_p"):
        training_args.top_p = getattr(script_args, "top_p", 1.0)
    if not hasattr(training_args, "top_k"):
        training_args.top_k = getattr(script_args, "top_k", 10)
    if not hasattr(training_args, "repetition_penalty"):
        training_args.repetition_penalty = getattr(script_args, "repetition_penalty", 1.0)

    print("training config:", training_args)
    print("script config:", script_args)

    trainer = GRPOTrainerV2(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        data_collator=collator_instance
    )

    #trainer.train(resume_from_checkpoint=True)
    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
