import numpy as np
from dataset.imports import import_json
from tools import struct, to_structs, concat_lists, to_dicts, pluck, pprint_struct

from dataset.annotate import decode_obj

import operator
from functools import reduce

import math

import json

def load_dataset(filename):
    return to_structs(import_json(filename))

def filter_used(dataset):
    return  [image for image in dataset.images if image.category in ['train', 'validate', 'test']]


def get_category(dataset, category):
    return  [image for image in dataset.images if image.category == category]



def set_category(category):
    def f(image):
        return image._extend(category = category)
    return f

def set_category_all(images, category):
    return list(map(set_category(category), images))


def quantiles(xs):
    return np.percentile(np.array(xs), [0, 25, 50, 75, 100])


def annotation_summary(dataset):
    images = filter_used(dataset)

    def count(image):
        annotations = [decode_obj(ann) for ann in image.annotations]
        annotations = [obj for obj in annotations if obj is not None]

        n = len(annotations)
        categories = struct (
            test = n if image.category == 'test' else 0,
            validate = n if image.category == 'validate' else 0,
            train = n if image.category == 'train' else 0,
        )


        def box_area(ann):
            x1, y1, x2, y2 = ann.box
            area = (x2 - x1) * (y2 - y1)
            return math.sqrt(area)


        def box_length(ann):
            x1, y1, x2, y2 = ann.box
            return max(x2 - x1, y2 - y1)

        box_areas = list(map(box_area, annotations))
        box_lengths = list(map(box_length, annotations))

        return struct(n = n, categories=categories, box_areas=box_areas, box_lengths=box_lengths)
    
    infos = list(map(count, images))
    totals = reduce(operator.add, infos)

    return struct(n_images = len(infos), categories = totals.categories, 
        n_annotations = totals.n, 
        n = quantiles(pluck('n', infos)), 
        box_length=quantiles(totals.box_lengths), 
        box_area = quantiles(totals.box_areas))

# pprint_struct(datasets._map(summary))


def combine_penguins(datasets):
    val_royd = set_category_all(get_category(datasets.royd, 'validate'), 'test_royd')
    val_hallett = set_category_all(get_category(datasets.hallett, 'validate'), 'test_hallett')
    val_cotter = set_category_all(get_category(datasets.cotter, 'validate'), 'test_cotter')

    train = list(concat_lists([get_category(datasets.royd, 'train'), 
        get_category(datasets.cotter, 'train'), 
        get_category(datasets.hallett, 'train')]))   

    combined = struct(config=datasets.royd.config, images=concat_lists([train, val_royd, val_hallett, val_cotter]))

    with open(combined_file, 'w') as outfile:
        json.dump(to_dicts(combined), outfile, sort_keys=True, indent=4, separators=(',', ': '))



def decode_dataset(data):
    data = to_structs(data)

    config = data.config
    classes = [struct(id = int(k), name = v) for k, v in config.classes.items()]

    images = filter_none([decode_image(i, config) for i in data.images])
    images.sort(key = lambda image: image.start)

    return struct(classes = classes, images = images)
