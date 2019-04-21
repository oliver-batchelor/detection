from .coco import import_coco
from .voc import import_voc

from tools.parameters import get_choice
from tools import struct
from dataset.annotate import decode_dataset

import json

def import_json(filename):
    with open(filename, "r") as file:
        dataset = json.loads(file.read())

        # Convert keys of classes to integers (json objects are always string)
        config = dataset['config']
        config['classes'] = {int(k):v for k, v in config['classes'].items()}

        return dataset

    raise Exception('load_file: file not readable ' + filename)

import_voc = import_voc
import_coco = import_coco



def lookup_classes(dataset, classes):
    config = dataset['config']
    class_map = {v['name'] : int(k) for k, v in config['classes'].items()}

    def lookup_class(name):
        assert name in class_map, "class not found: " + name
        return class_map[name]

    return list(map(lookup_class, classes))


def filter_annotations(image, class_ids):
    anns = {k:ann for k, ann in image['annotations'].items() if ann['label'] in class_ids}
    return {**image, 'annotations':anns}


def subset_dataset(dataset, subset, keep_classes=None):
    keep_classes = keep_classes or subset
        
    subset_ids = lookup_classes(dataset, subset)
    keep_ids = lookup_classes(dataset, keep_classes)

    assert all(c in subset_ids for c in keep_ids)

    keep_image = contains_any_class(dataset, subset_ids)
    images = [filter_annotations(image, keep_ids) 
                for image in dataset['images'] 
                if keep_image(image)
             ]
    
    class_config = dataset['config']['classes']
    subset = {k : class_config[k] for k in keep_ids}
    config = {**dataset['config'], 'classes':subset}

    return {'images': images, 'config': config}
    

def contains_any_class(dataset, class_ids):
    def f(image):
        for ann in image['annotations'].values():

            if ann['label'] in class_ids:
                return True
        return False
    return f

def add_dict(d, k):
    d[k] = d[k] + 1 if k in d else 1
    return d

def summarise(images, classes):
    counts = sum([len(image['annotations']) for image in images])

    categories = {}
    for image in images:
        add_dict(categories, image['category'])

    annotated = categories.get('train', 0) + categories.get('test', 0) + categories.get('validate', 0)

    print("using {:d} classes, found {:d} images, {:d} annotated with {:d} instances at {:.2f} per image"
        .format(len(classes), len(images), annotated, counts, counts / annotated) )
    
    print(categories)


def import_dataset(input_args, subset=None):
    choice, params = get_choice(input_args)
  

    if choice == 'json':
        print("loading json from: " + params.path)
        return import_json(params.path)
    elif choice == 'coco':
        print("loading coco from: " + params.path)
        return import_coco(params.path, classes=subset)
    elif choice == 'voc':
        print("loading voc from: " + params.path)        
        return import_voc(params.path, preset=params.preset)
    else:
        assert False, "unknown dataset type: " + choice

def load_dataset(args):

    subset = args.subset.split(",") if args.subset else None
    keep_classes = args.keep_classes.split(",") if args.keep_classes else subset

    dataset = import_dataset(args.input, subset)

    if subset:
        dataset = subset_dataset(dataset, subset, keep_classes)

    summarise(dataset['images'], dataset['config']['classes'])
  
    return decode_dataset(dataset)

