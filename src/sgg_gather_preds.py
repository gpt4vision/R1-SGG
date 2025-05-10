import os
import sys
import json
import glob
import torch
from tqdm import tqdm
import re

from datasets import load_dataset

import numpy as np
from scipy.optimize import linear_sum_assignment
import spacy

from utils.wordnet import find_synonym_map

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from src.vg_synonyms import obj2vg_list, rel2vg_list, pass2act_list, VG150_OBJ_CATEGORIES, VG150_PREDICATES, obj2psg_list


# Load spaCy model (with word vectors)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


DEBUG=os.getenv("DEBUG", "false").lower() == "true"



# Cache for spaCy docs to avoid repeated computations
doc_cache = {}

def get_doc(word):
    if word not in doc_cache:
        doc_cache[word] = nlp(word)
    return doc_cache[word]

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

def category_semantic_similarity(pred_id, gt_id):
    # Extract category names from ids (substring before the dot)
    cat_pred = pred_id.split('.')[0]
    cat_gt = gt_id.split('.')[0]
    doc_pred = get_doc(cat_pred)
    doc_gt = get_doc(cat_gt)
    return doc_pred.similarity(doc_gt)

def box_L1(boxA, boxB, im_size):
    iw, ih = im_size
    # Calculate the sum of absolute differences between the coordinates
    boxA = [boxA[0] / iw, boxA[1] / ih, boxA[2] / iw, boxA[3] / ih]
    boxB = [boxB[0] / iw, boxB[1] / ih, boxB[2] / iw, boxB[3] / ih]

    l1_distance = sum(abs(a - b) for a, b in zip(boxA, boxB))
    return l1_distance

def cost_function(pred, gt, im_size, sem_weight=0.5, iou_weight=0.5):
    assert len(pred['bbox']) == 4, "len(pred['bbox'])={}".format(len(pred['bbox']))

    iou = compute_iou(pred['bbox'], gt['bbox'])
    sem_sim = category_semantic_similarity(pred['id'], gt['id'])
    return sem_weight * (1.0 - sem_sim) + iou_weight * (1.0 - iou) + box_L1(pred['bbox'], gt['bbox'], im_size)

def bi_match(groundtruths, predictions, im_size):
    num_gt = len(groundtruths)
    num_pred = len(predictions)
    pad = max(0, num_gt - num_pred)
    cost_matrix = np.zeros((num_pred + pad, num_gt))
    
    # Fill in cost for each prediction-groundtruth pair
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(groundtruths):
            cost_matrix[i, j] = cost_function(pred, gt, im_size=im_size)
    if pad > 0:
        cost_matrix[num_pred:, :] = 10000  # Assign a high cost for padded rows

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
    obj = obj.replace("_", " ").replace("-", " ")
    return obj.strip().lower()

def scale_box(box, scale):
    sw, sh = scale
    return [box[0]*sw, box[1]*sh, box[2]*sw, box[3]*sh]

def is_box(item):
    return (
        isinstance(item, (list, tuple)) and
        len(item) == 4 and
        all(isinstance(e, (int, float)) for e in item)
    )


def visualize_assignments(image, pred_objs, gt_objs, assignments, filename,
                          pred_rels=None, gt_rels=None):
    from PIL import ImageDraw, ImageFont
    import matplotlib.pyplot as plt

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    w, h = image.size
    image_left = image.copy()
    image_right = image.copy()
    draw_left = ImageDraw.Draw(image_left)
    draw_right = ImageDraw.Draw(image_right)

    # Draw predictions (left image)
    for obj in pred_objs:
        bbox = obj['bbox']
        label = obj['id']
        draw_left.rectangle(bbox, outline='red', width=4)
        text_w, text_h = draw_left.textbbox((0, 0), label, font=font)[2:]
        text_bg = [bbox[0], bbox[1] - text_h, bbox[0] + text_w, bbox[1]]
        if text_bg[1] < 0:
            text_bg[1] = bbox[1]
            text_bg[3] = bbox[1] + text_h
        draw_left.rectangle(text_bg, fill='red')
        draw_left.text((text_bg[0], text_bg[1]), label, fill='white', font=font)

    # Draw groundtruths (right image)
    for obj in gt_objs:
        bbox = obj['bbox']
        label = obj['id']
        draw_right.rectangle(bbox, outline='green', width=4)
        text_w, text_h = draw_right.textbbox((0, 0), label, font=font)[2:]
        text_bg = [bbox[0], bbox[1] - text_h, bbox[0] + text_w, bbox[1]]
        if text_bg[1] < 0:
            text_bg[1] = bbox[1]
            text_bg[3] = bbox[1] + text_h
        draw_right.rectangle(text_bg, fill='green')
        draw_right.text((text_bg[0], text_bg[1]), label, fill='white', font=font)

    # Combine side-by-side
    combined_img = Image.new('RGB', (w * 2, h), (255, 255, 255))
    combined_img.paste(image_left, (0, 0))
    combined_img.paste(image_right, (w, 0))

    # Draw match lines and costs
    draw_combined = ImageDraw.Draw(combined_img)
    for match in assignments:
        gt_bbox = match['groundtruth']['bbox']
        gt_center = ((gt_bbox[0] + gt_bbox[2]) / 2, (gt_bbox[1] + gt_bbox[3]) / 2)
        gt_center_combined = (gt_center[0] + w, gt_center[1])
        cost = match['cost']
        if match['prediction'] is not None:
            pred_bbox = match['prediction']['bbox']
            pred_center = ((pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2)
            draw_combined.line([pred_center, gt_center_combined], fill='blue', width=3)
            cost_text = f"{cost:.3f}"
            mid_point = ((pred_center[0] + gt_center_combined[0]) / 2,
                         (pred_center[1] + gt_center_combined[1]) / 2)
            cost_bbox = draw_combined.textbbox((0, 0), cost_text, font=font)
            cost_w = cost_bbox[2] - cost_bbox[0]
            cost_h = cost_bbox[3] - cost_bbox[1]
            cost_bg = [mid_point[0] - cost_w / 2, mid_point[1] - cost_h / 2,
                       mid_point[0] + cost_w / 2, mid_point[1] + cost_h / 2]
            draw_combined.rectangle(cost_bg, fill='blue')
            draw_combined.text((cost_bg[0], cost_bg[1]), cost_text, fill='white', font=font)

    # Convert to matplotlib figure
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.imshow(combined_img)
    ax.set_title("Predictions (left) and Groundtruth (right) with Links")
    ax.axis('off')

    # Format relationships as text
    def format_rels(rels, title):
        lines = [f"{title}:"]
        for rel in rels:
            try:
                sub = rel['subject']
                pred = rel['predicate']
                obj = rel['object']
                lines.append(f"{sub} --{pred}--> {obj}")
            except:
                continue
        return "\n".join(lines) if len(lines) > 1 else ""

    pred_text = format_rels(pred_rels or [], "Predicted Relationships")
    gt_text = format_rels(gt_rels or [], "Ground Truth Relationships")

    full_text = pred_text + "\n\n" + gt_text
    plt.figtext(0.5, 0.01, full_text, wrap=True, ha='center', fontsize=10, fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()    

def main():
    is_psg = sys.argv[1] == 'psg'
    json_folder = sys.argv[2] 
    print("is_psg:", is_psg)
    if is_psg:
        psg_categories = json.load(open("src/psg_categories.json"))
        PSG_OBJ_CATEGORIES = psg_categories['thing_classes'] + psg_categories['stuff_classes']
        PSG_OBJ_CATEGORIES = [refine_node_edge(e) for e in PSG_OBJ_CATEGORIES] # remove '-', '_' 
        PSG_PREDICATES = psg_categories['predicate_classes']
        NAME2CAT = {name: idx for idx, name in enumerate(PSG_OBJ_CATEGORIES)}
        print("PSG_OBJ_CATEGORIES:\n", PSG_OBJ_CATEGORIES, "\n")    
    else:
        NAME2CAT = {name: idx for idx, name in enumerate(VG150_OBJ_CATEGORIES) if name != "__background__"}


    pred_files = glob.glob(os.path.join(json_folder, "*json"))
    preds = []
    for file_path in tqdm(pred_files, desc="Loading predictions"):
        with open(file_path, 'r') as f:
            preds.append(json.load(f))
            
    if not is_psg:
        db_raw = load_dataset("JosephZ/vg150_val_sgg_prompt")['train']
    else:
        db_raw = load_dataset("JosephZ/psg_test_sg")['train']
    db = {str(e['image_id']): e for e in tqdm(db_raw, desc="Loading dataset")}


    fails = [0, 0]
    preds_dict = {}
    cats = []
    if not is_psg:
        map_obj2target = {e['source']: e['target'] for e in obj2vg_list}
        map_rel2target = {e['source']: e['target'] for e in rel2vg_list}
    else:
        map_obj2target = {e['source']: e['target'] for e in obj2psg_list}
        map_rel2target = {}

    pass2act = {e['source']: e for e in pass2act_list}
    rel_cats = []
    obj_cats = []

    target_predicates = VG150_PREDICATES if not is_psg else ['__background__'] + PSG_PREDICATES
    target_objects = VG150_OBJ_CATEGORIES[1:] if not is_psg else PSG_PREDICATES

    all_im_ids = []
    for kk, item in enumerate(tqdm(preds, desc="Processing images")):
        if DEBUG and kk > 10:
            break

        im_id = item['image_id']
        all_im_ids.append(im_id)
    
        gt_objs = json.loads(item['gt_objects'])
        gt_rels = json.loads(item['gt_relationships'])
        fails[1] += 1

        resp = item['response']
        try:
            image = db[str(im_id)]['image']
            iw, ih = image.size
            box_scale = item['box_scale'] if "box_scale" in item else [1000.0, 1000.0] # Qwen2-VL use a normalization [0, 1000]
            scale_factors = (iw / box_scale[0], ih / box_scale[1])

            # Remove <answer> tags and parse JSON
            resp = extract_answer_content(resp)
            resp = json.loads(resp)

            pred_objs = resp['objects']
            pred_rels = resp['relationships']
            pred_objs_ = []
            for obj in pred_objs:
                assert len(obj['bbox']) == 4, "len(obj['bbox']) != 4"
                assert is_box(obj['bbox']), "invalid box :{}".format(obj['bbox'])
                assert 'id' in obj, "invalid obj:{}".format(obj)
                assert isinstance(obj['id'], str), f"invalid obj:{obj}"
                obj['bbox'] = scale_box(obj['bbox'], scale_factors)

                pred_objs_.append({'id': refine_node_edge(obj['id']), 'bbox': obj['bbox']})
            pred_objs = pred_objs_

            pred_rels_ = []
            for rel in pred_rels:
                tmp = {'subject': refine_node_edge(rel['subject']),
                       'predicate': refine_node_edge(rel['predicate']),
                       'object': refine_node_edge(rel['object']),
                      }
                pred_rels_.append(tmp)
            pred_rels = pred_rels_

        except Exception as e: 
            print(f"Fail to extract objs. & rels. of im_id:{im_id} from response:", item['response'], " Exception:", e)
            fails[0] += 1
            continue

        pred_objs_dict = {"image_id": im_id, "boxes": [], "labels": [], "scores": [], "names": [], "names_target":[]}
        for e in pred_objs:
            org_name = e['id']
            cat = org_name.split('.')[0]
            cats.append(cat)
            if cat not in target_objects:
                if cat not in map_obj2target:
                    db_ = find_synonym_map([cat], target_objects)
                    map_obj2target.update(db_)
                if cat in map_obj2target:
                    cat = map_obj2target[cat] 

            if cat in NAME2CAT:
                pred_objs_dict['labels'].append(NAME2CAT[cat])
                pred_objs_dict['boxes'].append(e['bbox'])
                pred_objs_dict['scores'].append(1.0)
                pred_objs_dict['names'].append(org_name)
                pred_objs_dict['names_target'].append(cat)

        if len(pred_objs_dict['boxes']) > 0:
            preds_dict[im_id] = pred_objs_dict
        else:
            continue

        # process relationships
        names = pred_objs_dict['names']
        names_set = set(names)
        all_node_pairs = []
        all_relation = []
        relation_tuples = []
        for rel in pred_rels:
            try:
                sub = rel['subject']
                obj = rel['object']
                predicate = rel['predicate']
                assert isinstance(sub, str) and isinstance(obj, str)
                rel_cats.append(predicate)
                obj_cats.append(sub.split('.')[0])
                obj_cats.append(obj.split('.')[0])
            except:
                continue

            if predicate not in target_predicates and predicate in pass2act:
                direction = pass2act[predicate]['passive']
                predicate = pass2act[predicate]['target']
                if direction == 1: # swap <subject, object>
                    sub_ = sub
                    sub = obj
                    obj = sub_ 
                

            if sub in names_set and obj in names_set:
                sid = names.index(sub)
                oid = names.index(obj)

                if (predicate not in target_predicates) and (predicate not in map_rel2target):
                    rel_syn = find_synonym_map([predicate], target_predicates[1:])
                    map_rel2target.update(rel_syn)

                if (predicate not in target_predicates) and (predicate in map_rel2target):
                    predicate = map_rel2target[predicate]
                   
                if predicate in target_predicates:
                    relation_tuples.append([pred_objs_dict['names_target'][sid], 
                                            pred_objs_dict['boxes'][sid], 
                                            pred_objs_dict['names_target'][oid],
                                            pred_objs_dict['boxes'][oid],
                                            predicate])

                    triplet = [sid, oid, 
                               target_predicates.index(predicate) 
                              ]
                    all_node_pairs.append([sid, oid])

                    tmp = [0]*len(target_predicates)
                    tmp[triplet[-1]] = 1
                    all_relation.append(tmp)

        preds_dict[im_id]['relation_tuples'] = relation_tuples
        preds_dict[im_id]['graph'] = {'all_node_pairs': all_node_pairs, 
                                      'all_relation': all_relation,
                                      'pred_boxes': pred_objs_dict['boxes'],
                                      'pred_boxes_class': pred_objs_dict['labels'],
                                      'pred_boxes_score': pred_objs_dict['scores']
                                      }

        if DEBUG:
            assignments = bi_match(gt_objs, pred_objs, (iw, ih) )
            for match in assignments:
                gt_id = match['groundtruth']['id']
                pred_id = match['prediction']['id'] if match['prediction'] is not None else "null"
                print(f"Groundtruth {gt_id} -> Prediction {pred_id} with cost {match['cost']:.3f}")
            
            visualize_assignments(image, pred_objs, gt_objs, assignments, f"rl-vis/{im_id}.jpg", pred_rels, gt_rels)

    cats = list(set(cats))
    print("fails:", fails)
    print("failure rate:", round(fails[0]/fails[1]*100.0, 4))
    print("Number of valid predictions:", len(preds_dict))
    for im_id in all_im_ids:
        pred_objs_dict = {"image_id": im_id, "boxes": torch.empty(1, 4).tolist(), "labels": [0], "scores": [0], "names": ["unknown"], "names_target":["unknown"]}
        if im_id not in preds_dict:
            preds_dict[im_id] = pred_objs_dict
            preds_dict[im_id]['relation_tuples'] = []
            preds_dict[im_id]['graph'] = {'all_node_pairs': torch.zeros(1, 2).long().tolist(),
                                          'all_relation': torch.zeros(1, len(target_predicates)).tolist(),
                                          'pred_boxes': pred_objs_dict['boxes'],
                                          'pred_boxes_class': pred_objs_dict['labels'],
                                          'pred_boxes_score': pred_objs_dict['scores']}


    rel_cats = list(set(rel_cats))
    print("rel_cats:", len(rel_cats), rel_cats)
    print("rel_cats (novel):", [e for e in rel_cats if e not in map_rel2target and e not in set(target_predicates)])
    obj_cats = list(set(obj_cats))
    print("obj_cats:", len(obj_cats), obj_cats)
    print("obj_cats (novel):", [e for e in obj_cats if e not in map_obj2target and e not in NAME2CAT])
    
    with open(sys.argv[3], 'w') as fout:
        json.dump(preds_dict, fout)

if __name__ == "__main__":
    main()    
