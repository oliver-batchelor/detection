
from os import path
import json

import torch
from dataset.detection import DetectionDataset


def load_dataset(filename):
    with open(filename, "r") as file:
        str = file.read()
        return decode_dataset()
    raise Exception('load_file: file not readable ' + filename)



def decode_dataset(data):
    config = data['config']

    classes = [{'id':int(k), 'name':v} for k, v in config['classes'].items()]
    class_mapping = {c['id']:i  for i, c in enumerate(classes)}

    def to_box(obj):
        b = obj['bounds']
        return [*b['lower'], *b['upper']]

    def to_label(obj):
        return class_mapping[obj['label']]

    def to_image(data):
        objs = [obj for obj in data['annotations']]

        boxes = [to_box(obj) for obj in objs]
        labels = [to_label(obj) for obj in objs]
        return {
            'file':path.join(config['root'], data['imageFile']),
            'boxes': torch.FloatTensor(boxes),
            'labels': torch.LongTensor(labels)
        }

    train = [to_image(i) for i in data['images'] if i['category'] == 'Train']
    test = [to_image(i) for i in data['images'] if i['category'] == 'Test']

    return DetectionDataset(classes=classes, train_images=train, test_images=test)
