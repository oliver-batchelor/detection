import numpy as np
from dataset.imports import import_json
from tools import struct, to_structs, concat_lists, to_dicts, pluck, pprint_struct, sum_list
import tools

from dataset.annotate import decode_obj
from detection.evaluate import match_targets

import operator
from functools import reduce
from datetime import datetime
import dateutil.parser as date

import math

import json
import os
from os import path

def load_dataset(filename):
    return to_structs(import_json(filename))

def filter_categories(dataset, categories=['train', 'validate', 'test']):
    return  [image for image in dataset.images if image.category in categories]



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

def image_annotations(image):
    annotations = [decode_obj(ann) for ann in image.annotations.values()]
    return [obj for obj in annotations if obj is not None]

def annotation_summary(dataset):
    def count(image):

        annotations = image_annotations(image)
        n = len(annotations)

        categories = struct (
            test = n if image.category == 'test' else 0,
            validate = n if image.category == 'validate' else 0,
            train = n if image.category == 'train' else 0,
            new = n if image.category == 'new' else 0,
            discard = n if image.category == 'discard' else 0,
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
    
    infos = list(map(count, filter_categories(dataset)))

    totals = reduce(operator.add, infos)
    return struct(n_images = len(infos), 
        categories = totals.categories, 
        n_annotations = totals.n, 
        n = quantiles(pluck('n', infos)), 
        box_length=quantiles(totals.box_lengths), 
        box_area = quantiles(totals.box_areas))


def match_datasets(dataset1, dataset2, threshold=0.5, check_overlap=True):

    images1 = {image.filename:image.target for image in dataset1.history}
    images2 = {image.filename:image.target for image in dataset2.history}

    results = {}
    for k, image1 in images1.items():
        image2 = images2.get(k)
        assert (not check_overlap) or image2 is not None, "image without match: " + k

        if image2 is not None:
            matches = match_targets(image1, image2, threshold=threshold)
        
        results[k] = struct(count1=image1._size, count2=image2._size, matched = len(matches))

    total = sum_list(list(results.values()))
    iou = total.matched / (total.count1 + total.count2 - total.matched)
    f1 = 2 * total.matched / (total.count1 + total.count2)

    return struct(total = total._extend(iou = iou, f1 = f1), images = results)


def decode_dataset(data):
    data = to_structs(data)

    config = data.config
    classes = [struct(id = int(k), name = v) for k, v in config.classes.items()]

    images = filter_none([decode_image(i, config) for i in data.images])
    images.sort(key = lambda image: image.start)

    return struct(classes = classes, images = images, config=config)

def image_date(filename):
    cmd = 'identify -format "%[EXIF:DateTimeOriginal]" ' + filename
    datestr = os.popen(cmd).read()
    
    return datetime.strptime(datestr.strip(), '%Y:%m:%d %H:%M:%S')


def class_annotations(image, class_id):
    return [ann for ann in image.annotations.values() if ann.label == class_id]
    

def weighted_total(image, config):
    return sum([config.classes[ann.label].count_weight for ann in image.annotations.values()], 0)


def get_counts(dataset, class_id=None):
    images = dataset.images

    def count(image):
        n =  weighted_total(image, dataset.config) if class_id is None else len(class_annotations(image, class_id))
        t = date.parse(image.image_creation)

        def f(entry):
            threshold, count = entry
            return count

        counts = image.detections.stats.counts if class_id is None \
            else image.detections.stats.class_counts[str(class_id)]._map(f)
 
        return struct(image_file = image.image_file, time = t, truth = n, \
             category = image.category, estimate = counts, error=abs(n - counts.middle))

    counts = list(map(count, images))
    return sorted(counts, key = lambda count: count.time)


# def iou_datasets(dataset1, dataset2, match_iou = 0.5):
