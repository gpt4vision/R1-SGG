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
import glob
import copy
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from collections import OrderedDict
from functools import partial

import torch
#import torch._dynamo
#torch._dynamo.config.suppress_errors = True

import numpy as np
import random
from datasets import load_dataset, load_from_disk
from transformers import AutoProcessor

from trainer import GRPOTrainerV2, GRPOConfig

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from scipy.optimize import linear_sum_assignment
import spacy
from functools import lru_cache


from open_r1.trainer.utils.misc import encode_image_to_base64

from trainer.utils.prompt_gallery import PROMPT_DET, PROMPT_CLS, OBJ_HOLDER, PROMPT_SG, PROMPT_SG_OPEN

# Set DEBUG_MODE flag and log path once
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
LOG_PATH = os.getenv("LOG_PATH", "debug.log")

STRICT_FORMAT = os.getenv("STRICT_FORMAT", "true").lower() == "true"

# Load spaCy model (with word vectors)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

SEM_WEIGHT = 1.0
IOU_WEIGHT = 2.0
BOX_L1_WEIGHT = 5.0

FORMAT_REWARD_WEIGHT = float(os.getenv("FORMAT_REWARD_WEIGHT", '1.0'))
NODE_REWARD_WEIGHT = float(os.getenv("NODE_REWARD_WEIGHT", '2.0'))
EDGE_REWARD_WEIGHT = float(os.getenv("EDGE_REWARD_WEIGHT", '5.0'))

VG150_BASE_OBJ_CATEGORIES = set(['tile', 'drawer', 'men', 'railing', 'stand', 'towel', 'sneaker', 'vegetable', 'screen', 'vehicle', 'animal', 'kite', 'cabinet', 'sink', 'wire', 'fruit', 'curtain', 'lamp', 'flag', 'pot', 'sock', 'boot', 'guy', 'kid', 'finger', 'basket', 'wave', 'lady', 'orange', 'number', 'toilet', 'post', 'room', 'paper', 'mountain', 'paw', 'banana', 'rock', 'cup', 'hill', 'house', 'airplane', 'plant', 'skier', 'fork', 'box', 'seat', 'engine', 'mouth', 'letter', 'windshield', 'desk', 'board', 'counter', 'branch', 'coat', 'logo', 'book', 'roof', 'tie', 'tower', 'glove', 'sheep', 'neck', 'shelf', 'bottle', 'cap', 'vase', 'racket', 'ski', 'phone', 'handle', 'boat', 'tire', 'flower', 'child', 'bowl', 'pillow', 'player', 'trunk', 'bag', 'wing', 'light', 'laptop', 'pizza', 'cow', 'truck', 'jean', 'eye', 'arm', 'leaf', 'bird', 'surfboard', 'umbrella', 'food', 'people', 'nose', 'beach', 'sidewalk', 'helmet', 'face', 'skateboard', 'motorcycle', 'clock', 'bear'])

VG150_BASE_PREDICATE = set(["between", "to", "made of", "looking at", "along", "laying on", "using", "carrying", "against", "mounted on", "sitting on", "flying in", "covering", "from", "over", "near", "hanging from", "across", "at", "above", "watching", "covered in", "wearing", "holding", "and", "standing on", "lying on", "growing on", "under", "on back of", "with", "has", "in front of", "behind", "parked on"])

VG150_OBJ_CATEGORIES = set(['airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra'])

VG150_PREDICATES = set(["above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near", "of", "on", "on back of", "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on", "standing on", "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"])




def answer_to_ovdr(example):
    objs = json.loads(example['objects'])
    rels = json.loads(example['relationships'])
    obj_map = {}
    idx = 1 
    new_objs = []
    for obj in objs:
        if obj['id'].split('.')[0] in VG150_BASE_OBJ_CATEGORIES:
            new_name = '%s.%s'%(obj['id'].split('.')[0], idx)
            obj_map[obj['id']] = new_name
            idx += 1
            tmp = {'id': new_name, 'bbox': obj['bbox']}
            new_objs.append(tmp)
    new_rels = []
    for rel in rels:
        if rel['predicate'] in VG150_BASE_PREDICATE and rel['subject'] in obj_map and rel['object'] in obj_map:
            tmp = {"subject": obj_map[rel['subject']], "predicate": rel['predicate'], "object": obj_map[rel['object']]}
            new_rels.append(tmp)

    if len(new_objs) == 0 or len(new_rels) == 0:
        return None  # mark for removal

    example['objects'] = json.dumps(new_objs)
    example['relationships'] = json.dumps(new_rels)
    return example
            
    

        


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """
    reward_funcs: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'."},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    use_predefined_cats: bool = field(
        default=False, 
        metadata={"help": "Whether to use predefined object categories"}
    )
    task_type: list[str] = field(
        default_factory=lambda: ["sgg"],
        metadata={"help": "List of tasks. Possible values: 'sgg', 'det', or 'cls'."}
    )
    use_think_prompt_inplace: bool = field(
        default=False,
        metadata={"help": "Whether to place <think>...</think> in the user's prompt."}
    )
    disable_think_tags: bool=field(
        default=False,
        metadata={"help": "Whether to disable <think> tags."}
    )
    use_ovdr_split: bool = field(
        default=False,
        metadata={"help": "Whether to use ovdr split for the dataset."}
    )
    use_fp8: bool=field(
        default=False, 
        metadata={"help": "Whether to use FP8 for training."}
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

def refine_node_edge(obj):
    obj = obj.replace('-merged', '').replace('-other', '')
    obj = obj.replace("_", " ").replace("-", " ")
    return obj.strip().lower()


@lru_cache(maxsize=4096)
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

def compute_giou(boxA, boxB):
    """
    Calculate the Generalized Intersection over Union (GIoU) of two bounding boxes.

    Parameters:
      boxA: list or tuple of [x1, y1, x2, y2]
      boxB: list or tuple of [x1, y1, x2, y2]

    Returns:
      giou: float, the Generalized IoU between boxA and boxB.
    """

    # Calculate the (x, y)-coordinates of the intersection rectangle.
    inter_x1 = max(boxA[0], boxB[0])
    inter_y1 = max(boxA[1], boxB[1])
    inter_x2 = min(boxA[2], boxB[2])
    inter_y2 = min(boxA[3], boxB[3])

    # Compute the width and height of the intersection rectangle.
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    # Compute the area of intersection rectangle.
    intersection = inter_width * inter_height

    # Compute the area of both bounding boxes.
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the union area.
    union = areaA + areaB - intersection
    # Ensure no division by zero.
    if union == 0:
        return 0.0

    # Compute the standard Intersection over Union (IoU).
    iou = intersection / union

    # Find the smallest (axis-aligned) box that encloses both boxes.
    c_x1 = min(boxA[0], boxB[0])
    c_y1 = min(boxA[1], boxB[1])
    c_x2 = max(boxA[2], boxB[2])
    c_y2 = max(boxA[3], boxB[3])
    areaC = (c_x2 - c_x1) * (c_y2 - c_y1)

    # Calculate the Generalized IoU.
    giou_value = iou - (areaC - union) / areaC

    return giou_value

def box_L1(boxA, boxB):
    # Calculate the sum of absolute differences between the coordinates
    l1_distance = sum(abs(a - b) for a, b in zip(boxA, boxB))
    return l1_distance    




def cost_function(pred, gt, sem_weight=SEM_WEIGHT, iou_weight=IOU_WEIGHT, box_l1_weight=BOX_L1_WEIGHT):
    assert len(pred['bbox']) == 4, f"Invalid bbox length: {len(pred['bbox'])}"

    iou = compute_giou(pred['bbox'], gt['bbox']) # use giou
    sem_sim = category_semantic_similarity(pred['id'], gt['id'])
    return sem_weight * (1.0 - sem_sim) + iou_weight * (1.0 - iou) + box_l1_weight * box_L1(pred['bbox'], gt['bbox'])


def _freeze_objs(objs):
    """
    Turn a list of scene‑graph objects into a hashable key:
    (‘id’, (x1, y1, x2, y2))
    """
    return tuple(
        (o["id"], tuple(o["bbox"])) for o in objs
    )

def _bi_match_impl(groundtruths, predictions,
                   sem_weight=SEM_WEIGHT,
                   iou_weight=IOU_WEIGHT,
                   box_l1_weight=BOX_L1_WEIGHT):
    num_gt = len(groundtruths)
    num_pred = len(predictions)
    pad = max(0, num_gt - num_pred)
    cost_matrix = np.zeros((num_pred + pad, num_gt))

    for i, pred in enumerate(predictions):
        for j, gt in enumerate(groundtruths):
            cost_matrix[i, j] = cost_function(
                pred, gt, sem_weight, iou_weight, box_l1_weight
            )

    if pad:
        cost_matrix[num_pred:, :] = 1e5       # high cost for padded rows

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return [
        {
            "groundtruth": groundtruths[c],
            "prediction":  predictions[r],
            "cost":        cost_matrix[r, c],
        }
        for r, c in zip(row_ind, col_ind)
        if r < num_pred
    ]

def bi_match_triplets(gt_rels, pred_rels):
    num_gt = len(gt_rels)
    num_pred = len(pred_rels)
    pad = max(0, num_gt - num_pred)
    cost_matrix = np.zeros((num_pred + pad, num_gt))
    for i, pred in enumerate(pred_rels):
        for j, gt in enumerate(gt_rels):
            cost_matrix[i, j] = 1.0 - category_semantic_similarity(refine_node_edge(pred["subject"]), refine_node_edge(gt["subject"]) ) *\
                                category_semantic_similarity(refine_node_edge(pred["object"]),  refine_node_edge(gt["object"]) ) *\
                                category_semantic_similarity(refine_node_edge(pred["predicate"]), refine_node_edge(gt["predicate"]))
        
    if pad:
        cost_matrix[num_pred:, :] = 1e5

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    return [
        {
            "groundtruth": gt_rels[c],
            "prediction":  pred_rels[r],
            "cost":        cost_matrix[r, c],
        }
        for r, c in zip(row_ind, col_ind)
        if r < num_pred
    ]
 


@lru_cache(maxsize=4096)
def _bi_match_cached(gt_key, pred_key,
                     sem_weight, iou_weight, box_l1_weight):
    """
    Hashable wrapper so we can cache across identical calls.
    Converts the frozen keys back to lists of dicts for the impl fn.
    """
    groundtruths = [
        {"id": obj_id, "bbox": list(bbox)}
        for obj_id, bbox in gt_key
    ]
    predictions = [
        {"id": obj_id, "bbox": list(bbox)}
        for obj_id, bbox in pred_key
    ]
    return _bi_match_impl(
        groundtruths, predictions,
        sem_weight, iou_weight, box_l1_weight
    )

def bi_match(groundtruths, predictions,
             sem_weight=SEM_WEIGHT,
             iou_weight=IOU_WEIGHT,
             box_l1_weight=BOX_L1_WEIGHT):
    """
    Thin, cached front‑end that keeps the original signature.
    """
    return _bi_match_cached(
        _freeze_objs(groundtruths),
        _freeze_objs(predictions),
        sem_weight, iou_weight, box_l1_weight,
    )

#def bi_match(groundtruths, predictions, sem_weight=SEM_WEIGHT, iou_weight=IOU_WEIGHT, box_l1_weight=BOX_L1_WEIGHT):
#    num_gt = len(groundtruths)
#    num_pred = len(predictions)
#    pad = max(0, num_gt - num_pred)
#    cost_matrix = np.zeros((num_pred + pad, num_gt))
#
#    for i, pred in enumerate(predictions):
#        for j, gt in enumerate(groundtruths):
#            cost_matrix[i, j] = cost_function(pred, gt, sem_weight, iou_weight, box_l1_weight)
#    if pad > 0:
#        cost_matrix[num_pred:, :] = 100000  # High cost for padded rows
#
#    row_ind, col_ind = linear_sum_assignment(cost_matrix)
#    assignments = []
#    for r, c in zip(row_ind, col_ind):
#        if r >= num_pred:
#            continue
#        assignments.append({
#            'groundtruth': groundtruths[c],
#            'prediction': predictions[r],
#            'cost': cost_matrix[r, c]
#        })
#    return assignments


def build_conversation(image, text, use_think_system_prompt=False):
#    system_prompt = SYSTEM_PROMPT if use_think_system_prompt else "You are a helpful and multimodal AI assistant."

    base64_image = encode_image_to_base64(image)
    messages = [
#        {
#            "role": "system",
#            "content": system_prompt
#        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": text},
            ],
        },
    ]
    return messages

def resize_longest_side(img, target_long=224):
    w, h = img.size
    if w > h:
        new_w = target_long
        new_h = int(h * (target_long / w))
    else:
        new_h = target_long
        new_w = int(w * (target_long / h))
    return img.resize((new_w, new_h))

def self_evaluation(completions, image_id, image, task_type_list, **kwargs):
    prompts = {}
    for completion, im_id, pil_image, task_type in zip(completions, image_id, image, task_type_list):
        content = completion[0]['content']
        try:
            answer = extract_answer_content(content)
            preds = json.loads(answer)
            pred_objs = preds['objects']
            pred_rels = preds['relationships']
            name2box = {obj['id']: obj['bbox'] for obj in pred_objs}
        except:
            continue

        try:
            for obj in pred_objs:
                obj_name = obj['id'].split('.')[0]
                box = obj['bbox']  # [x1, y1, x2, y2]
                roi = pil_image.crop(box)
                roi = resize_longest_side(roi, target_long=224)

                choices = ['no visible object', obj_name]
                neg_objs = list(VG150_OBJ_CATEGORIES - set([obj_name]))
                choices += random.sample(neg_objs, 2)
                random.shuffle(choices)

                object_query = f"Which object appears in the image? Choose the most appropriate candidate from A) {choices[0]}; B) {choices[1]}; C) {choices[2]}; D) {choices[3]}."
                object_query_prompt = build_conversation(roi, object_query)
                qid = 'object-' + str(im_id) + '-' + obj_name + str(obj['bbox'])

                prompts[qid] = object_query_prompt
        except:
            pass

        try:
            for rel in pred_rels:
                sub_box = name2box[rel['subject']]
                obj_box = name2box[rel['object']]
                sub_name = rel['subject'].split('.')[0]
                obj_name = rel['object'].split('.')[0]
                union_box = [
                    min(sub_box[0], obj_box[0]),
                    min(sub_box[1], obj_box[1]),
                    max(sub_box[2], obj_box[2]),
                    max(sub_box[3], obj_box[3])
                ]
                roi = pil_image.crop(union_box)
                roi = resize_longest_side(roi, target_long=512)

                choices = ['no visible relationship', rel['predicate']]
                neg_rels = list(set(VG150_PREDICATES) - set([rel['predicate']]))
                choices += random.sample(neg_rels, 2)
                random.shuffle(choices)

                rel_query = f"What is the relationship between object {sub_name} and object {obj_name}? Choose the most appropriate candidate from A) {choices[0]}; B) {choices[1]}; C) {choices[2]}; D) {choices[3]}."
                rel_query_prompt = build_conversation(roi, rel_query)
                qid = 'rel-' + str(im_id) + '-' + sub_name + '-' + \
                      str(sub_box) + '-' + obj_name + '-' + str(obj_box) + \
                      '-' + str(rel['predicate'])

                prompts[qid] = rel_query_prompt
        except:
            pass

    return prompts

        
def scale_box(box, scale):
    sw, sh = scale
    assert len(box) == 4, " len(box) != 4 "
    return [int(box[0]*sw), int(box[1]*sh), int(box[2]*sw), int(box[3]*sh)]



def node_acc_reward(completions, solution, image_id, task_type_list, box_scale, **kwargs):
    """Compute node-level rewards."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol, im_id, task_type, box_wh in zip(contents, solution, image_id, task_type_list, box_scale):
        reward = 0.0
        match_objects = []
        if task_type not in ['sgg', 'det']:
            rewards.append(0)
            continue
        try:
            gt_objs = sol['objects']
            preds = json.loads(extract_answer_content(content))
            pred_objs = preds['objects']
            _objs = []
            for obj in pred_objs:
                obj['bbox'] = scale_box(obj['bbox'], (1.0 / box_wh[0], 1.0 / box_wh[1]))
                obj['id'] = refine_node_edge(obj['id'])
                _objs.append(obj)
            pred_objs = _objs


            assignments = bi_match(gt_objs, pred_objs)
            for assign in assignments:
                gt_id = assign['groundtruth']['id']
                pred_entry = assign['prediction']
                if pred_entry is None or pred_entry.get('id') is None:
                    continue
                pred_id = pred_entry['id']
                match_objects.append(
                    f"Groundtruth {gt_id} -> Prediction {pred_id} with cost {assign['cost']:.3f}"
                )
                reward +=  category_semantic_similarity(gt_id, pred_id) * NODE_REWARD_WEIGHT

            reward /= len(gt_objs) if gt_objs else 1
        except Exception:
            reward = 0.0

        rewards.append(reward)
        if DEBUG_MODE:
            with open(LOG_PATH, "a") as f:
                f.write(f"------------- {current_time} task_type:{task_type} Node-level Acc. Reward {reward:.3f} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"image_id: {im_id}, solution: {sol}\n")
                if match_objects:
                    f.write(f"Match objects: {match_objects}\n")
    return rewards

def node_box_reward(completions, solution, image_id, task_type_list, box_scale, **kwargs):
    """Compute node-level rewards."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol, im_id, task_type, box_wh in zip(contents, solution, image_id, task_type_list, box_scale):
        reward = 0.0
        match_objects = []
        if task_type not in ['sgg', 'det']:
            rewards.append(0)
            continue
        try:
            gt_objs = sol['objects']
            preds = json.loads(extract_answer_content(content))
            pred_objs = preds['objects']
            _objs = []
            for obj in pred_objs:
                obj['bbox'] = scale_box(obj['bbox'], (1.0 / box_wh[0], 1.0 / box_wh[1]))
                obj['id'] = refine_node_edge(obj['id'])
                _objs.append(obj)
            pred_objs = _objs

            assignments = bi_match(gt_objs, pred_objs)
            for assign in assignments:
                gt_id = assign['groundtruth']['id']
                pred_entry = assign['prediction']
                if pred_entry is None or pred_entry.get('id') is None:
                    continue
                pred_id = pred_entry['id']
                match_objects.append(
                    f"Groundtruth {gt_id} -> Prediction {pred_id} with cost {assign['cost']:.3f}"
                )
                reward += (compute_iou(assign['groundtruth']['bbox'], pred_entry['bbox']) * IOU_WEIGHT + \
                          np.exp(-box_L1(assign['groundtruth']['bbox'], pred_entry['bbox'])) * BOX_L1_WEIGHT) / (IOU_WEIGHT+BOX_L1_WEIGHT) * NODE_REWARD_WEIGHT

            reward /= len(gt_objs) if gt_objs else 1
        except Exception:
            reward = 0.0

        rewards.append(reward)
        if DEBUG_MODE:
            with open(LOG_PATH, "a") as f:
                f.write(f"------------- {current_time} task_type:{task_type} Node-level IoU Reward {reward:.3f} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"image_id: {im_id}, solution: {sol}\n")
                if match_objects:
                    f.write(f"Match objects: {match_objects}\n")
    return rewards


def edge_reward(completions, solution, image_id,  task_type_list, box_scale, **kwargs):
    """Compute edge-level rewards."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol, im_id, task_type, box_wh in zip(contents, solution, image_id, task_type_list, box_scale):
        reward = 0.0
        match_objects = []
        match_triplets = []
        if task_type not in ['sgg', 'cls']:
            rewards.append(0)
            continue
        try:
            preds = json.loads(extract_answer_content(content))
            if task_type == 'sgg':
                gt_objs = sol['objects']
                gt_rels = sol['relationships']
                pred_objs = preds['objects']
                pred_rels = preds['relationships']
                _objs = []
                for obj in pred_objs:
                    obj['bbox'] = scale_box(obj['bbox'], (1.0 / box_wh[0], 1.0 / box_wh[1]))
                    obj['id'] = refine_node_edge(obj['id'])
                    _objs.append(obj)
                pred_objs = _objs

                assignments = bi_match(gt_objs, pred_objs)
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

                pred_triplets = { (refine_node_edge(rel['subject']), refine_node_edge(rel['object'])): \
                                   refine_node_edge(rel['predicate']) for rel in pred_rels }

                gt_boxes = {e['id']: e['bbox'] for e in gt_objs}
                pred_boxes = {e['id']: e['bbox'] for e in pred_objs}

                for gt_rel in gt_rels:
                    sub, obj = gt_rel['subject'], gt_rel['object']
                    if (sub not in map_obj) or (obj not in map_obj):
                        continue
                    sub_mapped = map_obj[sub]
                    obj_mapped = map_obj[obj]
                    if (sub_mapped, obj_mapped) in pred_triplets:
                        pred_pred = pred_triplets[(sub_mapped, obj_mapped)]
                        
                        reward += category_semantic_similarity(sub, sub_mapped) * \
                                  category_semantic_similarity(obj, obj_mapped) * \
                                  category_semantic_similarity(gt_rel['predicate'], pred_pred) * \
                                  EDGE_REWARD_WEIGHT

                        match_triplets.append(
                            f"GT triplet: {sub} -> {gt_rel['predicate']} -> {obj}, "
                            f"Pred: {sub_mapped} -> {pred_pred} -> {obj_mapped}"
                        )
                reward /= max(1, len(gt_rels))
            elif task_type == 'cls':
                assignments = bi_match_triplets(sol["relationships"], preds["relationships"])
                reward = 0
                for assign in assignments:
                    gt = assign["groundtruth"]
                    pred = assign["prediction"]

                    reward += category_semantic_similarity(refine_node_edge(pred["subject"]), refine_node_edge(gt["subject"])) * \
                              category_semantic_similarity(refine_node_edge(pred["object"]), refine_node_edge(gt["object"])) * \
                              category_semantic_similarity(refine_node_edge(pred["predicate"]), refine_node_edge(gt["predicate"])) * EDGE_REWARD_WEIGHT

                reward /= max(1, len(sol["relationships"])) 
        except Exception:
            reward = 0.0

        rewards.append(reward)
        if DEBUG_MODE:
            with open(LOG_PATH, "a") as f:
                f.write(f"------------- {current_time} task_type:{task_type} Edge-level Reward {reward:.3f} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"image_id: {im_id}, solution: {sol}\n")
                if match_objects:
                    f.write(f"Match objects: {match_objects}\n")
                if match_triplets:
                    f.write(f"Match triplets: {match_triplets}\n")
    return rewards


def is_valid_box(item):
    if not isinstance(item, dict):
        return False

    bbox = item.get("bbox") 
    if "id" not in item or not isinstance(bbox, list) or len(bbox) != 4:
        return False

    return all(isinstance(e, (int, float)) for e in bbox)


def is_valid_predicate(item):
    if not isinstance(item, dict):
        return False

    keys = ("subject", "object", "predicate")
    if not all(k in item for k in keys):
        return False

    return all(isinstance(item[k], str) for k in keys)

def format_reward(completions, image_id, task_type_list, **kwargs):
    """
    Reward function that checks if the completion has the correct format:
    - Must contain <think>...</think> and <answer>...</answer>
    - The <answer> content must be a valid JSON dict with keys "objects" and "relationships"
    """
    pattern = r"<think>.*?</think>\s*<answer>(.*?)</answer>"

    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for completion, im_id, task_type in zip(completions, image_id, task_type_list):
        content = completion[0]["content"].strip() 
        match = re.fullmatch(pattern, content, re.DOTALL)
        reward = 0.0
        try:
            answer_json = json.loads(extract_answer_content(content))
            if task_type == 'sgg':
                if isinstance(answer_json, dict) and ("objects" in answer_json) and ("relationships" in answer_json):
                    objs = set([e['id'] for e in answer_json["objects"]])
                    graph_valid = True
                    for obj in answer_json["objects"]:
                        if not is_valid_box(obj):
                            graph_valid = False
                            break
                    for rel in answer_json["relationships"]:
                        sub = rel["subject"]
                        obj = rel["object"]
                        if (sub not in objs) or (obj not in objs) or (not is_valid_predicate(rel)):
                            graph_valid = False
                            break
                    
                    if match and graph_valid:
                        reward = 1.0
                    elif graph_valid and (not STRICT_FORMAT):
                        reward = 0.5 
                    else:
                        reward = 0.0
            elif task_type == 'det':
                if isinstance(answer_json, dict) and ("objects" in answer_json):
                    obj_valid = True
                    for obj in answer_json["objects"]:
                        if not is_valid_box(obj):
                            obj_valid = False
                            break
                    if match and obj_valid:
                        reward = 1.0
                    elif obj_valid and (not STRICT_FORMAT):
                        reward = 0.5
                    else: 
                        reward = 0.0
            elif task_type == 'cls':
                if isinstance(answer_json, dict) and ("relationships" in answer_json):
                    rel_valid = True
                    for rel in answer_json["relationships"]:
                        if not is_valid_predicate(rel):
                            rel_valid = False
                            break
                    if match and rel_valid:
                        reward = 1.0
                    elif rel_valid and (not STRICT_FORMAT):
                        reward = 0.5
                    else:
                        reward = 0.0 
            
            rewards.append(reward * FORMAT_REWARD_WEIGHT)
        except Exception:
            rewards.append(0.0)

        if DEBUG_MODE:
            with open(LOG_PATH, "a") as f:
                f.write(f"------------- {current_time} task_type:{task_type} Format Reward {reward:.3f} -------------\n")
                f.write(f"image_id:{im_id}, content: {content}\n")

    return rewards


# Reward functions registry
reward_funcs_registry = {
    "format_reward": format_reward,
    "node_acc_reward": node_acc_reward,
    "node_box_reward": node_box_reward,
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
    if script_args.reward_funcs is None:
        script_args.reward_funcs = ['format_reward', 'node_acc_reward', "node_box_reward",  "edge_reward"]

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset = load_dataset(script_args.dataset_name)['train']

    def assign_task_type(example, task_pool, rng):
        example["task_type"] = rng.choice(task_pool)
        return example
    
    rng = random.Random(training_args.seed)  
    dataset = dataset.map(partial(assign_task_type, task_pool=script_args.task_type, rng=rng))
    if script_args.use_ovdr_split:
        dataset = dataset.map(answer_to_ovdr, remove_columns=[], load_from_cache_file=False)
        dataset = dataset.filter(lambda x: x is not None)
    print("len(dataset):", len(dataset), "with task_type:", script_args.task_type)

    def replace_answer_format(item: str) -> str:
        return item.replace("<answer>", "```json").replace("</answer>", "```")

    class Collator(object):
        def __init__(self, processor, model_type,
                     use_predefined_cats,
                     use_think_prompt_inplace=False, 
                     disable_think_tags=False,
                     ):
            self.processor = processor
            self.model_type = model_type
            self.use_predefined_cats = use_predefined_cats
            self.use_think_prompt_inplace = use_think_prompt_inplace
            self.disable_think_tags = disable_think_tags

        def __repr__(self):
            return (
                f"{self.__class__.__name__}(\n"
                f"  model_type={self.model_type},\n"
                f"  processor={self.processor.__class__.__name__},\n"
                f"  use_predefined_cats={self.use_predefined_cats},\n"
                f"  use_think_prompt_inplace={self.use_think_prompt_inplace},\n"
                f"  disable_think_tags={self.disable_think_tags},\n"
                f")"
            )            

        def __call__(self, examples):
            batch = []
            images, prompts = [], []
            for example in examples:
                image = example["image"].convert('RGB')
                org_iw, org_ih = image.size
                images.append(image)
                if self.use_predefined_cats:
                    org_prompt = example['prompt_close'] #w. predefined categories
                else:
                    org_prompt = PROMPT_SG_OPEN

                org_prompt = org_prompt.replace(f"of size ({org_iw} x {org_ih}) ", "") # not provide image size
                org_prompt = replace_answer_format(org_prompt)
                #     
                if self.use_think_prompt_inplace:
                    org_prompt = PROMPT_SG

                system_prompt = "You are a helpful and multimodal AI assistant." if self.use_think_prompt_inplace or self.disable_think_tags else SYSTEM_PROMPT
                prompt = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", 
                             "text": org_prompt
                            }
                        ]
                    }
                ]
                prompts.append(prompt)

            if self.model_type == 'qwen2vl':
                input_width, input_height = [1000.0]*len(images), [1000.0]*len(images)
            elif self.model_type == 'qwen2.5vl':
                texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                         for msg in prompts
                ]
                image_inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt") 
                input_height = [image_inputs['image_grid_thw'][idx][1].item()*14 for idx in range(len(images))]
                input_width = [image_inputs['image_grid_thw'][idx][2].item()*14 for idx in range(len(images))]
            else:
                raise Exception("TODO")
            
            for image, example, prompt, input_iw, input_ih in zip(images, examples, prompts, input_width, input_height):
                org_iw, org_ih = image.size
                box_scale = [input_iw, input_ih]

                # load GTs.
                gt_objs = example["objects"]
                gt_rels = example["relationships"]
                if not isinstance(gt_objs, (list, tuple)):
                    gt_objs = json.loads(gt_objs)
                if not isinstance(gt_rels, (list, tuple)):
                    gt_rels = json.loads(gt_rels)

                new_objs = []
                # normalize box to [0,1]
                for obj in gt_objs:
                    obj['bbox'] = scale_box(obj['bbox'], (1.0/org_iw, 1.0/org_ih))
                    new_objs.append(obj)
                gt_objs = new_objs


                scene_graph = {"objects": gt_objs, "relationships": gt_rels}
                solution = scene_graph

                batch.append({"prompt": prompt, 
                              "image": image, 
                              "solution": solution,
                              "image_id": example['image_id'],
                              "task_type_list": "sgg",
                              "box_scale": box_scale,
                              })

            return batch

    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    except:
        rank = 0
        world_size = 1

    model_type=None
    base_name = None
      

    model_name = model_args.model_name_or_path.lower()
    
    if any(key in model_name for key in ['qwen2vl', 'qwen2-vl', 'qwen-2-vl']):
        model_type = "qwen2vl"
        if '7b' in model_name:
            base_name = "Qwen/Qwen2-VL-7B-Instruct"
        elif '2b' in model_name:
            base_name = "Qwen/Qwen2-VL-2B-Instruct"
        else:
            raise Exception(f"Unknown model size in: {model_name}")
    
    elif any(key in model_name for key in ['qwen2.5vl', 'qwen2.5-vl', 'qwen2-5-vl', 'qwen-2.5-vl']):
        model_type = "qwen2.5vl"
        if '7b' in model_name:
            base_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        elif '3b' in model_name:
            base_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        else:
            raise Exception(f"Unknown model size in: {model_name}")
    
    else:
        raise Exception(f"Unknown model type: {model_args.model_name_or_path}")


    processor = AutoProcessor.from_pretrained(base_name, 
                    min_pixels=script_args.min_pixels,
                    max_pixels=script_args.max_pixels)

    pad_token_id = processor.tokenizer.pad_token_id
    processor.pad_token_id = pad_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id

    collator_instance = Collator(processor,  model_type=model_type,
                                 use_predefined_cats=script_args.use_predefined_cats, 
                                 use_think_prompt_inplace=script_args.use_think_prompt_inplace,
                                 disable_think_tags=script_args.disable_think_tags,
                                )

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
        data_collator=collator_instance,
        processing_class=processor,
        model_type=model_type,
        use_fp8=script_args.use_fp8
    )
    # Check for existing checkpoint
    def find_valid_checkpoint(output_dir):
        ckpt_re = re.compile(r"checkpoint-(\d+)$")      # ↳ ends right after the digits
        
        checkpoints = sorted(
            [
                p for p in glob.glob(os.path.join(output_dir, "checkpoint-*"))
                if ckpt_re.search(os.path.basename(p))   # keep only pure-numeric checkpoints
            ],
            key=lambda p: int(ckpt_re.search(os.path.basename(p)).group(1))
        )
        for ckpt in reversed(checkpoints):  # Check latest first
            if glob.glob(os.path.join(ckpt, "global_step*")):
                return ckpt
        return None
    
    ckpt_to_resume = find_valid_checkpoint(training_args.output_dir)
    if ckpt_to_resume:
        print(f"[INFO] Resuming from checkpoint: {ckpt_to_resume}")
        trainer.train(resume_from_checkpoint=ckpt_to_resume)
    else:
        print("[INFO] Starting training from scratch")
        trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
