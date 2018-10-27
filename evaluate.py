import torch
import math
import gc

import torch.nn.functional as F
from torch.autograd import Variable

from tools.image import index_map
import tools.image.cv as cv

import tools.confusion as c

from tools.image.transforms import normalize_batch
from tools import Struct, tensor

import detection.box as box

from detection import evaluate


def to_device(t, device):
    if isinstance(t, list):
        return [to_device(x, device) for x in t]

    return t.to(device)


def eval_train(loss_func, device=torch.cuda.current_device()):

    def f(model, data):
        model.train()
        image, targets, lengths = data['image'], data['targets'], data['lengths']

        norm_data = to_device(normalize_batch(image), device)
        predictions = model(norm_data)

        class_loss, loc_loss, n = loss_func(to_device(targets, device), predictions)
        error = class_loss + loc_loss

        stats = Struct(error=error.item(), class_loss=class_loss.item(), loc_loss=loc_loss.item(), size=image.size(0), boxes=lengths.sum().item(), matches=n)
        return Struct(error=error, statistics=stats)

    return f

def summarize_train(name, stats, epoch, globals={}):
    avg_loss = stats.error / stats.size
    avg_loc = stats.loc_loss / stats.size
    avg_class = stats.class_loss / stats.size
    avg_matched = stats.matches / stats.size
    avg_boxes = stats.boxes / stats.size

    print(name + ' epoch: {}\tBoxes (truth, matches) {:.2f} {:.2f} \tLoss (class, loc, total): {:.6f}, {:.6f}, {:.6f}'.format(epoch, avg_boxes, avg_matched, avg_class, avg_loc, avg_loss))
    return avg_loss


def splits(size, n, overlap=0):
    div = (size + (n - 1) * overlap) / n

    prev = div
    ranges = [(0, round(div))]
    for i in range(n - 1):
        start = prev - overlap
        prev = start + div
        r = (round(start), round(prev))
        ranges.append(r)

    return ranges


def image_splits(size, n=(1, 1), overlap=0):
    w, h = size
    nx, ny = n

    return [((lx, ly), (ux, uy))
        for lx, ux in splits(w, nx, overlap)
        for ly, uy in splits(h, ny, overlap) ]


def split_image(image, n=(1, 1), overlap=0):

    def sub_image(ranges):
        (lx, ly), (ux, uy) = ranges
        return ((lx, ly), image.narrow(0, ly, uy - ly).narrow(1, lx, ux - lx))

    size = (image.size(1), image.size(0))
    return [ sub_image(r) for r in image_splits(size, n, overlap) ]



def split_sizes(min_splits, aspect):
    """
    Find the split sizes (how many divisions on each axis)
    to split an image, depending on aspect ratio.
    """

    max_aspect = max(aspect, 1/aspect)
    minor, major = 1, 1

    while minor * major < min_splits:
        if major / minor <= pow(max_aspect, 1.8):
            major = major + 1
        else:
            minor = minor + 1
            major = minor

    if aspect >= 1:
        return major, minor
    else:
        return minor, major



def find_split_config(image, max_pixels=None):
    pixels = image.size(1) * image.size(0)

    if max_pixels is not None:
        min_splits = math.ceil(max_pixels / pixels)
        return split_sizes(min_splits, image.size(0) / image.size(1))

    return (1, 1)


def evaluate_split(model, image, encoder, nms_params=box.nms_defaults, n=(1, 1), overlap=0, device=torch.cuda.current_device()):
    model.eval()
    with torch.no_grad():
        splits = split_image(image, n, overlap)

        outputs = [evaluate_decode(model, image, encoder, device=device, offset=offset) for offset, image in splits]
        boxes, labels, confs = zip(*outputs)

        return encoder.nms(torch.cat(boxes, 0), torch.cat(labels, 0), torch.cat(confs, 0), nms_params=nms_params)


def evaluate_image(model, image, encoder, nms_params=box.nms_defaults, device=torch.cuda.current_device()):
    model.eval()
    with torch.no_grad():
        boxes, labels, confs = evaluate_decode(model, image, encoder, device)
        return encoder.nms(boxes, labels, confs, nms_params=nms_params)

def evaluate_raw(model, image, device):
    if image.dim() == 3:
        image = image.unsqueeze(0)

    assert image.dim() == 4, "evaluate: expected image of 4d  [1,H,W,C] or 3d [H,W,C]"
    assert image.size(0) == 1, "evaluate: expected batch size of 1 for evaluation"

    norm_data = to_device(normalize_batch(image), device)
    loc_preds, class_preds = model(norm_data)

    loc_preds = loc_preds.detach()[0]
    class_preds = class_preds.detach()[0]

    gc.collect()
    return loc_preds, class_preds

def evaluate_decode(model, image, encoder, device, offset = (0, 0)):
    loc_preds, class_preds = evaluate_raw(model, image, device=device)
    boxes, labels, confs = encoder.decode(image, loc_preds, class_preds)

    offset = torch.Tensor([*offset, *offset]).type_as(loc_preds)
    return boxes + offset, labels, confs

def eval_test(encoder, nms_params=box.nms_defaults, device=torch.cuda.current_device()):

    def f(model, data):

        image, target_boxes, target_labels = data['image'], data['boxes'], data['labels']
        boxes, labels, confs = evaluate_image(model, image.squeeze(0), encoder, nms_params=nms_params, device=device)

        thresholds = [0.5 + inc * 0.05 for inc in range(0, 10)]
        scores = torch.FloatTensor(10).zero_()

        def mAP(iou):
            _, _, score = evaluate.mAP(boxes, labels, confs, target_boxes.type_as(boxes).squeeze(0), target_labels.type_as(labels).squeeze(0), threshold = iou)
            return score

        if(boxes.size(0) > 0):
            scores = torch.FloatTensor([mAP(t) for t in thresholds])


        stats = Struct(AP=scores.mean().item(), mAPs=scores, size=1)
        return Struct(statistics=stats)

    return f

def summarize_test(name, stats, epoch, globals={}):
    mAPs =' '.join(['{:.2f}'.format(mAP * 100.0) for mAP in stats.mAPs / stats.size])
    AP = stats.AP * 100.0 / stats.size
    print(name + ' epoch: {}\t AP: {:.2f}\t mAPs@[0.5-0.95]: [{}]'.format(epoch, AP, mAPs))

    return AP
