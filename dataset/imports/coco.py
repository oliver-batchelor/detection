import os

import shutil
import argparse
import torch
import json

from pycocotools.coco import COCO
import numpy as np

from tools.image import cv, index_map
# import imports.voc as voc

from tools import struct, concat_lists
from tools.image.index_map import default_colors

from dataset.annotate import decode_dataset

classes = \
    ['person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']


# coco_mapping = {'aeroplane':'airplane', 'diningtable':'dining table', 'motorbike':'motorcycle', 'sofa':'couch', 'tv/monitor':'tv'}
#
# def to_coco(name):
#     return coco_mapping[name] if name in coco_mapping else name

# def load_coco(filename):
#     with open(filename, "r") as file:
#         str = file.read()
#         return decode_dataset(json.loads(str))
#     raise Exception('load_file: file not readable ' + filename)





def tagged(tag, contents):
    return {'tag':tag, 'contents':contents}

def import_subset(input, subset, target_category='Train', class_inputs=None):
    ann_file = '%s/annotations/instances_%s.json'%(input, subset)

    coco=COCO(ann_file)

    cat_ids = coco.getCatIds(class_inputs) if class_inputs else coco.getCatIds()
    print("{} classes found".format(len(cat_ids)))

    cats = coco.loadCats(cat_ids)
    class_names = [cat['name'] for cat in cats]
    class_map = {
        cat['id']: {
            'name':cat['name'],
            'colour':default_colors[int(cat['id']) % 255],
            'shape':'box'
        } for cat in cats
    }

    if classes and (not len(classes) == len(class_names)):
         for name in class_inputs:
             assert name in class_names, "class not found: " + name


    image_ids = concat_lists([coco.getImgIds(catIds=[cat]) for cat in cat_ids])

    print("found images: ", len(image_ids))
    print(class_names)

    def convert(id):
        info = coco.loadImgs(id)[0]
        file_name = info['file_name']

        input_image = '%s/%s'%(subset, file_name)
        def import_ann(ann):
            x, y, w, h = ann['bbox']
            return {
              'label': ann['category_id'],
              'confirm': True,
              'detection': None,
              'shape': tagged('box', {'lower': [x, y], 'upper': [x + w, y + h]  })
            }

        anns = coco.loadAnns(coco.getAnnIds(id, catIds=cat_ids))
        annotations = {k:import_ann(ann) for k, ann in enumerate(anns)}

        return {
            'image_file':input_image,
            'image_size':[info['width'], info['height']],
            'category':target_category,
            'annotations':annotations
        }

    images = list(map(convert, image_ids))
    return {
        'config': {
            'root':input,
            'extensions':[".jpg"],
            'classes':class_map
        },
        'images':images
    }



def import_coco(input_path="/home/oliver/storage/coco", classes=None, 
        subsets = [('train2017', 'train'), ('val2017', 'test')]):   
    
    imports = {subset : import_subset(input_path,  subset = subset, target_category= category, class_inputs = classes) for subset, category in subsets}
    first = next(iter(imports.values()))

    return {
        'config' : first['config'],
        'images' : sum([subset['images'] for subset in imports.values()], [])
    }




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Microsoft COCO, import dataset')

    parser.add_argument('--input', default='/home/oliver/storage/coco',
                        help='input image path')

    parser.add_argument('--output', default=None, required=True,
                        help='convert dataset and output to path')


    parser.add_argument('--restrict', default=None,
                    help='restrict to classes (comma sep) when converting dataset')

    # parser.add_argument('--voc', action='store_true', default=False,
    #                     help='use voc subset of classes')


    args = parser.parse_args()


    classes = args.restrict.split(",") if args.restrict else None
    coco = import_coco(args.input, classes)

    with open(args.output, 'w') as outfile:
        json.dump(all, outfile, sort_keys=True, indent=4, separators=(',', ': '))
