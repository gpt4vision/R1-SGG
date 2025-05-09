from tqdm import tqdm
import numpy as np
import copy

import torch
from pycocotools.coco import COCO

from src.utils.sgg_eval import SggEvaluator


VG150_OBJ_CATEGORIES = ['__background__', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

VG150_PREDICATES = ['__background__', "above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near", "of", "on", "on back of", "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on", "standing on", "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"]


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


class MyDataset(object):
    def __init__(self, db, db_type='vg150'):
        self._coco = None
        self.db_type = db_type 
        assert self.db_type in ['vg150', 'psg']
        
        if self.db_type == 'vg150':
            self.ind_to_classes = VG150_OBJ_CATEGORIES
            self.ind_to_predicates = VG150_PREDICATES
            self.name2classes = {name: cls for cls, name in enumerate(self.ind_to_classes) if name != "__background__"}
            self.categories = [{'supercategory': 'none', # not used?
                                'id': idx, 
                                'name': self.ind_to_classes[idx]}  
                                for idx in range(len(self.ind_to_classes)) if self.ind_to_classes[idx] != '__background__'
                                ]
        elif self.db_type == 'psg':
            psg_categories = json.load(open("src/psg_categories.json"))
            PSG_OBJ_CATEGORIES = psg_categories['thing_classes'] + psg_categories['stuff_classes']
            PSG_PREDICATES = psg_categories['predicate_classes']
            self.ind_to_classes = PSG_OBJ_CATEGORIES
            self.ind_to_predicates = ['__background__'] + PSG_PREDICATES
            self.name2classes = {name: cls for cls, name in enumerate(self.ind_to_classes) if name != "__background__"}
            self.categories = [{'supercategory': 'none', # not used?
                                'id': idx, 
                                'name': self.ind_to_classes[idx]}  
                                for idx in range(len(self.ind_to_classes)) if self.ind_to_classes[idx] != '__background__'
                                ]
            

        self.images = []
        self.annotations = []
        self.ids = []
        for item in tqdm(db):
            im_id = item['image_id']

            self.images.append({'id': im_id})
            self.ids.append(im_id)
            objs = json.loads(item['objects'])

            ann = {'image_id': im_id, 'labels': [], 'boxes': []}
            names = []
            for obj in objs:
                name, box = obj['id'].split('.')[0], obj['bbox']
                names.append(obj['id'])
                cls = self.name2classes[name]
                ann['labels'].append(cls)
                ann['boxes'].append(box)
            
            rels = json.loads(item['relationships'])
            edges = []
            for rel in rels:
                sub = rel['subject']
                obj = rel['object']
                pred = rel['predicate']
                sid = names.index(sub)
                oid = names.index(obj)
                tmp = [sid, oid, self.ind_to_predicates.index(pred)]
                edges.append(tmp)

            ann['edges'] = edges
            self.annotations.append(ann)

        print("total images", len(self.images), self.images[0])

    def get_groundtruth(self, index):
        ann = self.annotations[index]

        return torch.as_tensor(ann['boxes']), \
               torch.as_tensor(ann['labels']), \
               torch.as_tensor(ann['edges'])



    @property
    def coco(self):
        if self._coco is None:
            _coco = COCO()
            coco_dicts = dict(
                            images=self.images, 
                            annotations=[],
                            categories=self.categories)
            
            for ann in tqdm(self.annotations):
                for cls, box in zip(ann['labels'], ann['boxes']):
                    assert len(box) == 4
                    item = {
                            'area': (box[3] - box[1]) * (box[2] - box[0]),
                            'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]], # xywh
                            'category_id': cls,
                            'image_id': ann['image_id'], 
                            'id': len(coco_dicts['annotations']),
                            'iscrowd': 0,
                           }                    
                    coco_dicts['annotations'].append(item)

            _coco.dataset = coco_dicts
            _coco.createIndex()
            self._coco = _coco

        return self._coco

def refine_node_edge(obj):
    """ remove speical chars in the name. """
    obj = obj.replace("_", " ").replace("-", " ")
    return obj.strip().lower()

if __name__ == "__main__":
    import os
    import sys
    import json
    from datasets import load_dataset
    import torch
    from collections import defaultdict

    preds = json.load(open(sys.argv[2]))
    db = load_dataset(sys.argv[1])['train']
    db_type = 'vg150' if 'psg' not in sys.argv[1] else 'psg'
    dataset = MyDataset(db, db_type)


    ngR = []
    mR = defaultdict(list)
    ngR_per_image = []
    mR_per_image = defaultdict(list)
    num_gt_rels = 0
    for gt in tqdm(db):
        im_id = gt['image_id']
        if im_id in preds: # to prevent wrong generated image_id
            pred = preds[im_id]
        else:
            pred = None
        gt_rels = json.loads(gt['relationships'])
        gt_objects = json.loads(gt['objects'])
        gt_boxes = {refine_node_edge(obj['id']): obj['bbox'] for obj in gt_objects}
        recall = []
        recall_per_cat = defaultdict(list)

        for gt_rel in gt_rels:
             num_gt_rels += 1
             match = False
             gt_pred = refine_node_edge(gt_rel['predicate'])
             gt_sub_name = refine_node_edge(gt_rel['subject'])
             gt_obj_name = refine_node_edge(gt_rel['object'])

             if pred is not None:
                 for pred_rel in pred['relation_tuples']:
                     if refine_node_edge(gt_pred) != refine_node_edge(pred_rel[-1]):
                         continue

                     if gt_sub_name.split('.')[0].strip() != refine_node_edge(pred_rel[0]).split('.')[0].strip() or \
                        gt_obj_name.split('.')[0].strip() != refine_node_edge(pred_rel[2]).split('.')[0].strip():
                         continue

                     sub_iou = compute_iou(gt_boxes[gt_sub_name], pred_rel[1])
                     obj_iou = compute_iou(gt_boxes[gt_obj_name],  pred_rel[3])
                     if sub_iou >= 0.5 and obj_iou >= 0.5:
                         match = True
                         break

             recall.append(match)
             ngR.append(match)
             mR[gt_pred].append(match)
             recall_per_cat[gt_pred].append(match)

        if len(recall) > 0: 
             ngR_per_image.append(sum(recall) / len(recall) )

        for k in recall_per_cat.keys():
             mR_per_image[k].append(sum(recall_per_cat[k]) / len(recall_per_cat[k]) )
        
    mR_list = []
    for k in mR.keys():
        tmp = round(np.mean(mR[k]), 4)
        mR_list.append((k, tmp))
                 
    ngR_per_image = np.mean(ngR_per_image)
    mR_per_image = [(cat, round(np.mean(mR_per_image[cat]), 4) ) for cat in mR_per_image.keys()]


    sgg_evaluator = SggEvaluator(dataset, iou_types=("bbox","relation"), 
                                 num_workers=4, 
                                 num_rel_category=len(dataset.ind_to_predicates))

    def to_torch(item):
        for k in item.keys():
            try:
                item[k] = torch.as_tensor(item[k])
            except:
                pass


    k0 = None
    for k in tqdm(preds.keys()):
        k0 = k
        to_torch(preds[k])
        if 'graph' in preds[k]:
            graph = preds[k]['graph']
            to_torch(graph)
            preds[k]['graph'] = graph

    print('id:', k0, ' v:', preds[k0])

    sgg_evaluator.update(preds)
    sgg_evaluator.synchronize_between_processes()

    sgg_res = sgg_evaluator.accumulate()
    sgg_evaluator.summarize()
    sgg_evaluator.reset()


    print("whole ng recall list:", mR_list)
    print(f'whole ngR: {np.mean(ngR) * 100:.2f}')
    print(f'whole mean of ngR: {sum([e[1] for e in mR_list]) / len(mR_list) * 100:.2f}')
    print(f'ngR per image:{ngR_per_image * 100:.2f}')
    print(f'mean ngR per image: {sum([e[1] for e in mR_per_image]) / len(mR_per_image) * 100:.2f}')
    print(f'mean ngR list:{mR_per_image}')
