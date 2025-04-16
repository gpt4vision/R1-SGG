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

VG150_OBJ_CATEGORIES = ['__background__', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

NAME2CAT = {name: idx for idx, name in enumerate(VG150_OBJ_CATEGORIES) if name != "__background__"}

VG150_PREDICATES = ['__background__', "above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near", "of", "on", "on back of", "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on", "standing on", "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"]

# Load spaCy model (with word vectors)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


DEBUG=False

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
    return obj.replace("_", " ").replace("-", " ").lower()

def scale_box(box, scale):
    sw, sh = scale
    return [int(box[0]*sw), int(box[1]*sh), int(box[2]*sw), int(box[3]*sh)]

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
    json_folder = sys.argv[1] 

    pred_files = glob.glob(os.path.join(json_folder, "*json"))
    preds = []
    for file_path in tqdm(pred_files, desc="Loading predictions"):
        with open(file_path, 'r') as f:
            preds.append(json.load(f))
            
    is_qwen2vl = True # normalize bbox to [0, 1000]
    print("is_qwen2vl:", is_qwen2vl)
    if is_qwen2vl:
        # Load the dataset and convert to a dict for faster lookup.
        db_raw = load_dataset("JosephZ/vg150_val_sgg_prompt")['train']
        db = {e['image_id']: e for e in tqdm(db_raw, desc="Loading dataset")}

    pass2act = json.load(open("src/pass2act.json"))
    pass2act = {e['source']: e for e in pass2act}

    fails = [0, 0]
    preds_dict = {}
    cats = []
    # Load the pre-existing synonym mapping
    with open('src/synonym_mapping.json', 'r') as f:
        map_db = json.load(f)
    map_db_rel = {}
    rel_cats = []

    all_im_ids = []
    for kk, item in enumerate(tqdm(preds, desc="Processing images")):
        if DEBUG and kk > 10:
            break

        im_id = item['image_id']
        all_im_ids.append(im_id)
        image = db[im_id]['image']
        if is_qwen2vl: # for Qwen2VL, the output is normalized to [0, 1000]
            iw, ih = image.size
            scale_factors = (iw / 1000.0, ih / 1000.0)
            # Uncomment the following line to log image details if necessary
            # print("image id:", im_id, " image size:", image.size)
    
        gt_objs = json.loads(item['gt_objects'])
        gt_rels = json.loads(item['gt_relationships'])
        fails[1] += 1

        resp = item['response']
        try:
            # Remove <answer> tags and parse JSON
            resp = extract_answer_content(resp)
            resp = json.loads(resp)

            pred_objs = resp['objects']
            pred_rels = resp['relationships']
            for obj in pred_objs:
                assert len(obj['bbox']) == 4, "len(obj['bbox']) != 4"
                assert is_box(obj['bbox']), "invalid box :{}".format(obj['bbox'])
                assert 'id' in obj, "invalid obj:{}".format(obj)

            new_pred_rels = []
            for rel in pred_rels:
                if isinstance(rel, str) and '->' in rel:
                    tmp = rel.split('->')
                    try:
                        assert len(tmp) == 3, "CHECK relationship triplet:{}".format(tmp)
                        new_pred_rels.append({"subject": tmp[0].strip(), "predicate": tmp[1].strip(), "object": tmp[2].strip()})
                    except:
                        continue

            if isinstance(pred_rels[0], str):
                pred_rels = new_pred_rels

        except Exception as e:
            print(f"Fail to extract objs. & rels. of im_id:{im_id} from response:", item['response'])
            fails[0] += 1
            continue

        pred_objs_dict = {"image_id": im_id, "boxes": [], "labels": [], "scores": [], "names": []}
        for e in pred_objs:
            if is_qwen2vl:
                e['bbox'] = scale_box(e['bbox'], scale_factors)

            org_name = e['id']
            cat = org_name.split('.')[0].replace('-', ' ').replace('_', ' ').lower()
            cats.append(cat)
            if cat not in map_db:
                # Cache new synonym mapping if missing
                db_ = find_synonym_map([cat], VG150_OBJ_CATEGORIES[1:])
                if len(db_) > 0:
                    print("Add a mapping:", db_)
                map_db.update(db_)
            if cat in map_db:
                pred_objs_dict['labels'].append(NAME2CAT[map_db[cat]])
                pred_objs_dict['boxes'].append(e['bbox'])
                pred_objs_dict['scores'].append(1.0)
                pred_objs_dict['names'].append(org_name)

        if len(pred_objs_dict['boxes']) > 0:
            preds_dict[im_id] = pred_objs_dict
        else:
            continue

        # Process relationships using a set for faster membership check.
        names = pred_objs_dict['names']
        names_set = set(names)
        all_node_pairs = []
        all_relation = []
        for rel in pred_rels:
            try:
                sub = rel['subject']
                obj = rel['object']
                predicate = rel['predicate'].replace('_', ' ').replace('-', ' ').strip().lower()
            except:
                continue

            if predicate in pass2act:
                direction = pass2act[predicate]['passive']
                predicate = pass2act[predicate]['target']
                if direction == 1: # swap <subject, object>
                    sub_ = sub
                    sub = obj
                    obj = sub_ 

            if sub in names_set and obj in names_set:
                sid = names.index(sub)
                oid = names.index(obj)
                if predicate not in map_db_rel:
                    map_db_rel.update(find_synonym_map([predicate], VG150_PREDICATES[1:]))
                rel_cats.append(predicate)
                if predicate in map_db_rel:
                    new_predicate = map_db_rel[predicate]
                    triplet = [sid, oid, VG150_PREDICATES.index(new_predicate)]
                    all_node_pairs.append([sid, oid])

                    tmp = [0]*len(VG150_OBJ_CATEGORIES)
                    tmp[triplet[-1]] = 1
                    all_relation.append(tmp)

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
    print("failure rate:", fails[0]/fails[1]*100.0)
    print("Number of valid predictions:", len(preds_dict))
    for im_id in all_im_ids:
        pred_objs_dict = {"image_id": im_id, "boxes": torch.randn(1, 4).tolist(), "labels": [0], "scores": [0], "names": ["unknown"]}
        if im_id not in preds_dict:
            preds_dict[im_id] = pred_objs_dict
            preds_dict[im_id]['graph'] = {'all_node_pairs': torch.zeros(1, 2).long().tolist(),
                                          'all_relation': torch.zeros(1, 51).tolist(),
                                          'pred_boxes': pred_objs_dict['boxes'],
                                          'pred_boxes_class': pred_objs_dict['labels'],
                                          'pred_boxes_score': pred_objs_dict['scores']}


    rel_cats = list(set(rel_cats))
    print("rel_cats:", len(rel_cats), rel_cats)
    with open(sys.argv[2], 'w') as fout:
        json.dump(preds_dict, fout)

if __name__ == "__main__":
    main()    
