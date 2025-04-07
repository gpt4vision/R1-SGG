from tqdm import tqdm
import numpy as np
import copy

from pycocotools.coco import COCO

from utils.sgg_eval import SggEvaluator


VG150_OBJ_CATEGORIES = ['__background__', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

VG150_PREDICATES = ['__background__', "above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near", "of", "on", "on back of", "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on", "standing on", "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"]


NAME2CAT = {name: idx for idx, name in enumerate(VG150_OBJ_CATEGORIES) if name != "__background__"}


class MyDataset(object):
    def __init__(self, db, keep_ids):
        self._coco = None
        
        self.ind_to_classes = VG150_OBJ_CATEGORIES
        self.ind_to_predicates = VG150_PREDICATES
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


if __name__ == "__main__":
    import os
    import sys
    import json
    from datasets import load_dataset
    import torch

    preds = json.load(open(sys.argv[1]))
    db = load_dataset("JosephZ/vg150_val_sgg_prompt")['train']
    keep_ids = preds.keys()

    dataset = MyDataset(db, keep_ids)

    sgg_evaluator = SggEvaluator(dataset, iou_types=("bbox","relation"), num_workers=4)

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
