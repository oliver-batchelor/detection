import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from detection import box
from tools import struct, const, pluck
import numpy as np

def bookend(*xs, dim=0):
    def to_tensor(xs):
        return xs if torch.is_tensor(xs) else torch.FloatTensor([xs])
    return torch.cat([to_tensor(x) for x in xs], dim)

def max_1d(t):
    assert t.dim() == 1
    x, i = t.max(0)
    return x.item(), i.item()


def max_2d(t):
    assert (t.dim() == 2)
    values, inds = t.max(0)
    value, i = values.max(0)
    j = inds[i]
    return value.item(), (i.item(), j.item())

def take_max_2d(t):
    v, (i, j) = max_2d(t)
    t[:, i] = 0
    t[j, :] = 0
    return v, (i, j)

def match_targets(target1, target2, threshold=0.5):
    ious = box.iou_matrix(target1.bbox, target2.bbox)
    matches = []

    if ious.size(0) == 0 or ious.size(1) == 0:
        return []

    value, loc = take_max_2d(ious)
    while value > threshold:
        matches.append(struct(match=loc, iou=value))
        value, loc = take_max_2d(ious)

    return matches



def match_boxes(prediction, target,  threshold=0.5, eps=1e-7):
    n = prediction.label.size(0)
    matches = []

    ious = box.iou_matrix(prediction.bbox, target.bbox)

    for i, p in enumerate(prediction._sequence()):
        match = None
        if ious.size(1) > 0:
            iou, j = max_1d(ious[i])
            
            label = target.label[j]
            matches_box = iou > threshold
            
            if matches_box:
                ious[:, j] = 0  # mark target overlaps to 0 so they won't be selected twice

                if p.label == label:
                    match = (j, iou)

        matches.append(p._extend(match = match))
    return matches



# def rev_cummax(v):
#     for i in range(v.size(0) - 1, 0, -1):
#         v[i - 1] = max(v[i - 1], v[i])

#     return v

def rev_cummax(v):
    flipped = v.flip(0).numpy()
    rev_max = np.maximum.accumulate(flipped)
    return torch.from_numpy(rev_max).flip(0)

def area_under_curve(xs, ys):
    i = (xs[1:] != xs[:-1]).nonzero().squeeze(1)
    return ((xs[i + 1] - xs[i]) * ys[i + 1]).sum().item()



def compute_mAP(matches, confidence, num_target, weight=1, eps=1e-7):

    false_positives = ((1 - matches) * weight).cumsum(0)
    true_positives = (matches * weight).cumsum(0)

    recall = true_positives / (num_target if num_target > 0 else 1)
    precision = true_positives / (true_positives + false_positives).clamp(min = eps)

    recall = bookend(0.0, recall, 1.0)
    precision = rev_cummax(bookend(1.0, precision, 0.0))

    false_positives = bookend(0, false_positives)
    true_positives = bookend(0, true_positives)

    return struct(
        recall = recall, 
        precision = precision, 

        confidence = bookend(1.0, confidence, 0.0),

        false_positives = false_positives,
        true_positives = true_positives,

        false_negatives = num_target - true_positives,  
        n = num_target,

        mAP = area_under_curve(recall, precision))

def mAP_matches(matches, num_target, eps=1e-7):
    true_positives = torch.FloatTensor([0 if m.match is None else 1 for m in matches])
    return compute_mAP(true_positives, num_target, eps)


def _match_positives(labels_pred, labels_target, ious, threshold=0.5):
    n, m = labels_pred.size(0), labels_target.size(0)
    assert ious.size() == torch.Size([n, m]) 
    ious = ious.clone()

    matches = torch.FloatTensor(n).zero_()

    for i in range(0, n):
        iou, j = ious[i].max(0)
        iou = iou.item()

        if iou > threshold:
            ious[:, j] = 0  # mark target overlaps to 0 so they won't be selected twice
            if labels_pred[i] == labels_target[j]:
                matches[i] = 1

    return matches



def match_positives(pred, target):
    assert pred.label.dim() == 1 and target.label.dim() == 1
    n, m = pred._size, target._size

    if m == 0 or n == 0:
        return const(torch.FloatTensor(n).zero_())

    ious = box.iou_matrix(pred.bbox, target.bbox)
    return lambda threshold: _match_positives(pred.label, target.label, ious, threshold=threshold)

def list_subset(xs, inds):
    return [xs[i] for i in inds]


def mAP_subset(image_pairs, iou):
    all_matches =  [match_positives(i.prediction, i.target)(iou) for i in image_pairs]

    def f (inds):
        pairs = list_subset(image_pairs, inds)
        confidence    = torch.cat([i.prediction.confidence for i in pairs]).float()
        confidence, order = confidence.sort(0, descending=True)    

        matches = torch.cat(list_subset(all_matches, inds))[order]
        n = sum(i.target.label.size(0) for i in pairs)

        return compute_mAP(matches, confidence.cpu(), n)
    return f


def mAP_weighted(image_pairs):
    confidence    = torch.cat([i.prediction.confidence for i in image_pairs]).float()
    confidence, order = confidence.sort(0, descending=True)    

    matchers =  [match_positives(i.prediction, i.target) for i in image_pairs]
    predicted_label = torch.cat([i.prediction.label for i in image_pairs])[order]
    image_counts = torch.FloatTensor([i.target.label.size(0) for i in image_pairs])

    def f(threshold, image_weights):  
        matches = torch.cat([match(threshold) for match in matchers])[order]

        def eval_weight(weight):
            assert weight.size(0) == len(image_pairs)
            match_weights = torch.cat([torch.full([i.prediction._size], w) 
                for i, w in zip(image_pairs, weight) ])[order]

            num_targets = torch.dot(image_counts, weight)
            return compute_mAP(matches, confidence.cpu(), num_targets, weight=match_weights)

        return list(map(eval_weight, image_weights))
    return f


def gaussian_weights(xs, x_eval, sigma):
    dx2 = (x_eval.unsqueeze(1) - xs).pow(2)
    return torch.exp(-dx2 / (2*sigma*sigma)) / (math.sqrt(2*math.pi) * sigma)        


def mAP_smoothed(image_pairs, xs):
    assert len(image_pairs) == xs.size(0)
    weighted_mAP = mAP_weighted(image_pairs)

    def f(threshold, x_eval, sigma):
        weights = gaussian_weights(xs, x_eval, sigma)
        results = weighted_mAP(threshold, weights)
        return torch.Tensor(pluck('mAP', results))
    return f


def mAP_classes(image_pairs, num_classes):
    confidence    = torch.cat([i.detections.confidence for i in image_pairs]).float()
    confidence, order = confidence.sort(0, descending=True)    

    matchers =  [match_positives(i.detections, i.target) for i in image_pairs]

    predicted_label = torch.cat([i.detections.label for i in image_pairs])[order]
    target_label = torch.cat([i.target.label for i in image_pairs])

    num_targets = torch.bincount(target_label, minlength=num_classes)
    
    def f(threshold):      

        matches = torch.cat([match(threshold) for match in matchers])[order]
        def compute_class(i):
            inds = [(predicted_label == i).nonzero().squeeze(1)]
            return compute_mAP(matches[inds], confidence[inds].cpu(), num_targets[i].item())

        return struct(
            total = compute_mAP(matches, confidence.cpu(),  target_label.size(0)), 
            classes = [compute_class(i) for i in range(0, num_classes)]
        )

    return f

