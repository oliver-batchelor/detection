
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from detection import box
from tools import Struct

def cat(*xs, dim=0):
    def to_tensor(xs):
        return xs if torch.is_tensor(xs) else torch.FloatTensor([xs])
    return torch.cat([to_tensor(x) for x in xs], dim)




def match_boxes(prediction, target,  threshold=0.5, eps=1e-7):
    n = prediction.labels.size(0)
    matches = []

    for i in range(0, n):
        p = prediction.index_select(i)
        match = Struct(box = p.boxes, label = p.label, confidence = p.confidence, iou = 0, match = None)
        if ious.size(1) > 0:
            iou, j = map(Tensor.item, ious[i].max(0))
            
            label = target.labels[j]
            matches_box = iou > threshold
            
            if matches_box:
                ious[:, j] = 0  # mark target overlaps to 0 so they won't be selected twice

                if p.labels == label:
                    match = match.extend(iou = iou, match = j)

        matches.append(match)
    return matches


def rev_cummax(v):
    for i in range(v.size(0) - 1, 0, -1):
        v[i - 1] = max(v[i - 1], v[i])

    return v

def area_under_curve(xs, ys):
    i = (xs[1:] != xs[:-1]).nonzero().squeeze(1)

    return ((xs[i + 1] - xs[i]) * ys[i + 1]).sum().item()



def compute_mAP(true_positives, num_target, eps=1e-7):
    false_positives = (1 - true_positives).cumsum(0)
    true_positives = true_positives.cumsum(0)

    recall = true_positives / (num_target if num_target > 0 else 0)
    precision = true_positives / (true_positives + false_positives).clamp(min = eps)

    recall = cat(0.0, recall, 1.0)
    precision = rev_cummax(cat(1.0, precision, 0.0))

    return recall, precision, area_under_curve(recall, precision)

def mAP_matches(matches, num_target, eps=1e-7):
    true_positives = torch.FloatTensor([0 if m.match is None else 1 for m in matches])
    return compute_mAP(true_positives, num_target, eps)


def match_positives(pred, target, threshold=0.5, eps=1e-7):

    n = pred.labels.size(0)
    m = target.labels.size(0)

    if m == 0 or n == 0:
        return compute_mAP(torch.FloatTensor(n).zero_(), m, eps=1e-7)

    ious = box.iou(pred.boxes, target.boxes)
    true_positives = torch.FloatTensor(n).zero_()

    for i in range(0, n):
        iou, j = ious[i].max(0)
        iou = iou.item()

        label = target.labels[j.item()]

        if iou > threshold:
            ious[:, j] = 0  # mark target overlaps to 0 so they won't be selected twice

            if pred.labels[i].item() == label:
                true_positives[i] = 1

    return true_positives




def mAP(images, threshold=0.5, eps=1e-7):

    true_positives = torch.cat([match_positives(i.prediction, i.target, threshold) for i in images])
    confidence    = torch.cat([i.prediction.confidence for i in images])
    
    n = sum(i.target.labels.size(0) for i in images)

    confidence, order = confidence.sort(0, descending=True)
    true_positives = true_positives[order]


    return compute_mAP(true_positives, n, eps=1e-7)
