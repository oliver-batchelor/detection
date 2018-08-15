
from os import path
import json

import torch
from dataset.detection import DetectionDataset

from tools import filterMap, pluck, filterNone

def load_dataset(filename):
    with open(filename, "r") as file:
        str = file.read()
        return decode_dataset()
    raise Exception('load_file: file not readable ' + filename)




def split_tagged(tagged):
    return tagged['tag'], tagged['contents'] if 'contents' in tagged else None

def tagged(name, contents):
    if contents is None:
        return {'tag':name}
    else:
        return {'tag':name, 'contents':contents}

def decode_image(data, config):
    class_mapping = {int(k):i  for i, k in enumerate(config['classes'].keys())}

    def decode_obj(obj):
        tag, shape = split_tagged(obj['shape'])
        if tag == 'BoxShape':
            return {
                'label' : class_mapping[obj['label']],
                'box'   : [*shape['lower'], *shape['upper']]
            }
        else:
            # Ignore unsupported annotation for now
            return None

    objs = filterMap(decode_obj, data['annotations'])

    return {
        'file':path.join(config['root'], data['imageFile']),
        'boxes': torch.FloatTensor(pluck('box', objs)),
        'labels': torch.LongTensor(pluck('label', objs)),
        'category': data['category']
    }

def filterDict(d):
    return {k: v for k, v in d.items() if v is not None}

def decode_dataset(data):
    config = data['config']
    classes = [{'id':int(k), 'name':v} for k, v in config['classes'].items()]

    def imageCat(cat):
        return filterDict( { i['imageFile']:decode_image(i, config) for i in data['images'] if i['category'] == cat })

    return config, DetectionDataset(classes=classes, train_images=imageCat('Train'), test_images=imageCat('Test'))
