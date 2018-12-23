
from os import path
import json

import torch
from dataset.detection import DetectionDataset
from tools import struct, to_structs

from tools import filter_map, pluck, filter_none, struct, table

def load_dataset(filename):
    with open(filename, "r") as file:
        str = file.read()
        return decode_dataset(json.loads(str))
    raise Exception('load_file: file not readable ' + filename)



def split_tagged(tagged):
    return tagged.tag, tagged.contents if 'contents' in tagged else None

def tagged(name, contents):
    if contents is None:
        return struct(tag = name)
    else:
        return struct(tag = name, contents = contents)



def decode_obj(obj):
    tag, shape = split_tagged(obj.shape)

    if tag == 'BoxShape':
        return struct(
            label = obj.label, 
            box = [*shape.lower, *shape.upper])

    elif tag == 'CircleShape':
        x, y, r = *shape.centre, shape.radius

        return struct(
            label = obj.label, box = [x - r, y - r, x + r, y + r])
    else:
        # Ignore unsupported annotation for now
        return None


def decode_detection(det):
    obj = decode_obj(det)
    if obj is not None:
        obj = obj._extend(confidence = det.confidence)

    return obj

def lookup(mapping):
    def f(i):
        assert i in mapping
        return mapping[i]
    return f

def decode_detections(detections, class_mapping):
    objs = filter_map(decode_detection, detections)

    boxes = pluck('box', objs)
    labels = list(map(lookup(class_mapping), pluck('label', objs)))

    return table (bbox = torch.FloatTensor(boxes) if len(boxes) else torch.FloatTensor(0, 4),
                  label = torch.LongTensor(labels),
                  confidence = torch.LongTensor(pluck('confidence', objs))
        )


def decode_objects(data, class_mapping):
    objs = filter_map(decode_obj, data.annotations)

    boxes = pluck('box', objs)
    labels = list(map(lookup(class_mapping), pluck('label', objs)))

    return table (bbox = torch.FloatTensor(boxes) if len(boxes) else torch.FloatTensor(0, 4),
                  label = torch.LongTensor(pluck('label', objs)))


def class_mapping(config):
    return {int(k):i  for i, k in enumerate(config.classes.keys())}


def decode_image(data, config):
    target = decode_objects(data, class_mapping(config))

    return struct(
        file = path.join(config.root, data.imageFile),
        target = target,
        category = data.category
    )

def filterDict(d):
    return {k: v for k, v in d.items() if v is not None}


def decode_dataset(data):
    data = to_structs(data)   
    config = data.config
    classes = [struct(id = int(k), name = v) for k, v in config.classes.items()]

    def imageCat(cat):
        return filterDict( { i.imageFile:decode_image(i, config) for i in data.images if i.category == cat })

    return config, DetectionDataset(classes=classes, train_images=imageCat('Train'), test_images=imageCat('Test'))


def init_dataset(config):
    classes = [struct(id = int(k), name = v) for k, v in config.classes.items()]

    return config, DetectionDataset(classes=classes)