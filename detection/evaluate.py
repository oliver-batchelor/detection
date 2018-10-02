
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from detection import box
from tools import Struct

def cat(*xs, dim=0):
    def to_tensor(xs):
        return xs if torch.is_tensor(xs) else torch.FloatTensor([xs])
    return torch.cat([to_tensor(x) for x in xs], dim)

def rev_cummax(v):
    for i in range(v.size(0) - 1, 0, -1):
        v[i - 1] = max(v[i - 1], v[i])

    return v

def area_under_curve(xs, ys):
    i = (xs[1:] != xs[:-1]).nonzero().squeeze(1)

    return ((xs[i + 1] - xs[i]) * ys[i + 1]).sum().item()


def match_boxes(boxes_pred, labels_pred, confidence, boxes_target, labels_target,  threshold=0.5, eps=1e-7):

    n = boxes_pred.size(0)
    assert labels_pred.size(0) == n and confidence.size(0) == n

    m = labels_target.size(0)
    ious = None
    if m > 0 and n > 0:
        ious = box.iou(boxes_pred, boxes_target)

    matches = []

    for i in range(0, n):
        iou, j = 0, -1
        label = -1

        if ious is not None:
            iou, j = ious[i].max(0)
            iou = iou.item()
            j = j.item()

            label = labels_target[j]

        matches_box = iou > threshold
        is_match = labels_pred[i].item() == label and matches_box

        if matches_box:
            ious[:, j] = 0  # mark target overlaps to 0 so they won't be selected twice

        match = Struct(
            iou = iou,
            match = j if is_match else None,
            confidence = confidence[i].item(),
            label = labels_pred[i].item(),
            box = boxes_pred[i]
        )

        matches.append(match)

    return matches


def compute_mAP(true_positives, num_targets, eps=1e-7):
    false_positives = (1 - true_positives).cumsum(0)
    true_positives = true_positives.cumsum(0)

    recall = true_positives / (num_targets if num_targets > 0 else 0)
    precision = true_positives / (true_positives + false_positives).clamp(min = eps)

    recall = cat(0.0, recall, 1.0)
    precision = rev_cummax(cat(1.0, precision, 0.0))

    return recall, precision, area_under_curve(recall, precision)

def mAP_matches(matches, num_targets, eps=1e-7):
    true_positives = torch.FloatTensor([0 if m.match is None else 1 for m in matches])
    return compute_mAP(true_positives, num_targets, eps)


def mAP(boxes_pred, labels_pred, confidence, boxes_target, labels_target, threshold=0.5, eps=1e-7):

    n = boxes_pred.size(0)
    m = labels_target.size(0)

    assert labels_pred.size(0) == n and confidence.size(0) == n
    if m == 0 or n == 0:
        return compute_mAP(torch.FloatTensor(n).zero_(), m, eps=1e-7)

    ious = box.iou(boxes_pred, boxes_target)
    true_positives = torch.FloatTensor(n).zero_()

    for i in range(0, n):
        iou, j = ious[i].max(0)
        iou = iou.item()

        label = labels_target[j.item()]

        if iou > threshold:
            ious[:, j] = 0  # mark target overlaps to 0 so they won't be selected twice

            if labels_pred[i].item() == label:
                true_positives[i] = 1

    return compute_mAP(true_positives, m, eps=1e-7)
