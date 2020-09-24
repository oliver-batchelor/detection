import torch
from torch import Tensor
import math

import tools.confusion as c

from tools.image.transforms import normalize_batch
from tools import struct, shape

from detection import box, evaluate, detection_table

def encode_shape(box, class_config):
    lower, upper =  box[:2], box[2:]

    if class_config.shape == 'circle':

        centre = ((lower + upper) * 0.5).tolist()
        radius = ((upper - lower).sum().item() / 4)

        circle_shape = struct(centre = centre, radius = radius)
        return tagged('circle', circle_shape)

    elif class_config.shape == 'box':
        return tagged('box', struct (lower = lower.tolist(), upper = upper.tolist()))

    assert False, "unsupported shape config: " + class_config.shape


def weight_counts(weight):
    def f(counts):
       return counts[1] * weight

    return f


def get_counts(detections, classes, thresholds=None):

    def count(class_id, t):
        confidence = torch.FloatTensor([d.confidence for d in detections if d.label == class_id])
        levels = {k : (t, (confidence > t).sum().item()) for k, t in t.items()}
        return Struct(levels)

    if thresholds is None:
        thresholds = {c.id : struct(lower=0, middle=0, upper=0) for c in classes}

    class_map = {c.id: c for c in classes}

    class_counts = {k: count(k, t)  for k, t in thresholds.items()}
    counts = tools.sum_list([counts._map(weight_counts(class_map[k].count_weight)) for k, counts in class_counts.items()])

    return counts, class_counts

def make_detections(predictions, classes, thresholds, scale=1, network_id=None):
    def detection(p):
        object_class = classes[p.label]

        return struct (
            shape      =  encode_shape(p.bbox.cpu() / scale, object_class),
            label      =  object_class.id,
            confidence = p.confidence.item(),
            match = int(p.match) if 'match' in p else None
        )
    detections = list(map(detection, predictions))
    total_confidence = torch.FloatTensor([d.confidence for d in detections])

    def score(ds):
        return (total_confidence ** 2).sum().item()

    counts, class_counts = get_counts(detections, classes, thresholds)

    stats = struct (
        score   = score(detections),
        class_score = {c.id : score([d for d in detections if d.label == c.id]) for c in classes},
        counts = counts,
        class_counts =  class_counts,
        network_id = (env.run, env.epoch)
    ) 

    return struct(instances = detections, stats = stats)