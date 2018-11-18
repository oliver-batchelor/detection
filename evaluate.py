import torch
from torch import Tensor
import math
import gc

import torch.nn.functional as F
from torch.autograd import Variable

from tools.image import index_map
import tools.image.cv as cv

import tools.confusion as c

from tools.image.transforms import normalize_batch
from tools import Struct, tensor, show_shapes

import detection.box as box
from detection import evaluate
import operator
from functools import reduce


def eval_forward(device=torch.cuda.current_device()):

    def f(model, data):
        norm_data = normalize_batch(data.image).to(device)
        return model(norm_data)

    return f




def count_classes(label, num_classes):

    class_counts = (label + 1).view(-1).bincount(minlength = num_classes + 2)   
    
    return Struct(
        ignored  = class_counts[0].item(),
        negative = class_counts[1].item(),
        classes = class_counts[2:],

        positive = class_counts[2:].sum().item(),
        total = label.numel()
    )


def batch_stats(batch):
    assert(batch.dim() == 4 and batch.size(3) == 3)
  
    batch = batch.float().div_(255)
    flat = batch.view(-1, 3)  

    return batch.size(0) * Struct(mean=flat.mean(0).cpu(), std=flat.std(0).cpu())


def eval_stats(classes, device=torch.cuda.current_device()):
    def f(data):
        image = data.image.to(device)

        return Struct(
            image = batch_stats(image),
            boxes=data.lengths.sum().item(),
            box_counts=count_classes(data.encoding.classification.to(device), len(classes)),
            size = image.size(0)
        )
    return f

def mean_results(results):
    total = reduce(operator.add, results)
    return total / total.size, total


def summarize_stats(results, epoch, globals={}):
    avg = mean_results(results)
    counts = avg.box_counts
    
    print ("image: mean = {}, std = {}".format(str(avg.image.mean), str(avg.image.std)))
    print("instances: {:.2f}, anchors {:.2f}, anchors/instance {:.2f}, positive {:.2f},  ignored {:.2f}, negative {:.2f} "
        .format(avg.boxes, counts.total, counts.positive / avg.boxes, counts.positive, counts.ignored, counts.negative ))
    
    balances = counts.classes / counts.positive
    print("class balances: {}".format(str(balances.tolist())))


def eval_train(model, loss_func, device=torch.cuda.current_device()):

    def f(data):

        image = data.image.to(device)
        norm_data = normalize_batch(image)
        prediction = model(norm_data)

        loss = loss_func(data.encoding._map(Tensor.to, device), prediction)
        total = sum(loss.values())

        stats = Struct(error=total.item(), loss = loss._map(Tensor.item),
            size=data.image.size(0), 
            instances=data.lengths.sum().item(),
        )

        return Struct(error = total, statistics=stats, size = data.image.size(0))

    return f

def summarize_train(name, results, epoch, log):
    avg, totals = mean_results(results)
    loss_str = " + ".join(["({} : {:.6f})".format(k, v) for k, v in sorted(avg.loss.items())])

    print(name + ' epoch: {}\t (instances : {:.2f}) \tloss: {} = {:.6f}'
        .format(epoch, avg.instances, loss_str, avg.error))


    log.scalars("loss", avg.loss._extend(total = avg.error), global_step = epoch)
    
    # log.scalar("loss", avg.error)
    # for name, loss in avg.losses:
    #     log.scalar("loss/{}".format(name), loss)

    return avg.error


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


# def evaluate_split(model, image, encoder, nms_params=box.nms_defaults, n=(1, 1), overlap=0, device=torch.cuda.current_device()):
#     model.eval()
#     with torch.no_grad():
#         splits = split_image(image, n, overlap)

#         outputs = [evaluate_decode(model, image, encoder, device=device, offset=offset) for offset, image in splits]
#         boxes, label, confs = zip(*outputs)

#         return encoder.nms(torch.cat(boxes, 0), torch.cat(label, 0), torch.cat(confs, 0), nms_params=nms_params)


def evaluate_image(model, image, encoder, nms_params=box.nms_defaults, device=torch.cuda.current_device()):


    model.eval()
    with torch.no_grad():
        prediction = evaluate_decode(model, image, encoder, device)
        return  encoder.nms(prediction, nms_params=nms_params)



def evaluate_raw(model, image, device):
    if image.dim() == 3:
        image = image.unsqueeze(0)

    assert image.dim() == 4, "evaluate: expected image of 4d  [1,H,W,C] or 3d [H,W,C]"
    assert image.size(0) == 1, "evaluate: expected batch size of 1 for evaluation"

    def detach(p):
        return p.detach()[0]

    norm_data = normalize_batch(image).to(device)
    predictions = model(norm_data)._map(detach)

    #gc.collect()
    return predictions

def evaluate_decode(model, image, encoder, device, offset = (0, 0)):
    preds = evaluate_raw(model, image, device=device)
    p = encoder.decode(image, preds)

    offset = torch.Tensor([*offset, *offset]).to(device)
    return p._extend(bbox = p.bbox + offset)



def eval_test(model, encoder, nms_params=box.nms_defaults, device=torch.cuda.current_device()):

    def f(data):
        prediction = evaluate_image(model, data.image.squeeze(0), encoder, nms_params=nms_params, device=device)
        # target = data.target._map(Tensor.squeeze, 0)

        return Struct (
            file = data.file, 
            target = data.target._map(Tensor.to, device), 
            prediction = prediction, 
            size = data.image.size(0))
    return f


def percentiles(t, n=100):
    assert t.dim() == 1
    return torch.from_numpy(np.percentile(t.numpy(), np.arange(0, n)))


def AP(results):
    thresholds = [0.5 + inc * 0.05 for inc in range(0, 10)]

    compute_mAP = evaluate.mAP_at(results)
    mAP = [compute_mAP(t).mAP for t in thresholds]

    return Struct(
        mAP = mAP,
        AP = sum(mAP) / len(mAP)
    )
    

def summarize_test(name, results, epoch, log):
    summary = AP(results)
    mAP_strs =' '.join(['{:.2f}'.format(mAP * 100.0) for mAP in summary.mAP])

    print(name + ' epoch: {}\t AP: {:.2f}\t mAP@[0.5-0.95]: [{}]'.format(epoch, summary.AP * 100, mAP_strs))

    log.scalars(name, Struct(AP = summary.AP * 100.0, mAP50 = summary.mAP[0] * 100.0, mAP75 = summary.mAP[5] * 100.0), global_step = epoch)
    return summary.AP


