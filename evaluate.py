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
from tools import struct, tensor, show_shapes, cat_tables, show_shapes_info, \
    Histogram, ZipList, transpose_structs, transpose_lists, pluck, Struct, filter_none

import detection.box as box
from detection import evaluate
import operator
from functools import reduce


def eval_forward(model, device=torch.cuda.current_device()):

    def f(data):
        norm_data = normalize_batch(data.image).to(device)
        return model(norm_data)

    return f






def mean_results(results):
    total = reduce(operator.add, results)
    return total / total.size, total

def sum_results(results):
    return reduce(operator.add, results)



def summarize_stats(results, epoch, globals={}):
    avg = mean_results(results)
    counts = avg.box_counts
    

    print ("image: mean = {}, std = {}".format(str(avg.image.mean), str(avg.image.std)))
    print("instances: {:.2f}, anchors {:.2f}, anchors/instance {:.2f}, positive {:.2f},  ignored {:.2f}, negative {:.2f} "
        .format(avg.boxes, counts.total, counts.positive / avg.boxes, counts.positive, counts.ignored, counts.negative ))

    balances = counts.classes / counts.positive
    print("class balances: {}".format(str(balances.tolist())))


def train_statistics(data, loss, prediction, encoding, debug = struct(), device=torch.cuda.current_device()):
    num_classes = prediction.classification.size(2)

    stats = struct(error=loss.sum(),
        loss = loss._map(Tensor.item),
        size = data.image.size(0),
        instances=data.lengths.sum().item(),
        # class_instances=count_instances(data.target.label, num_classes)
    )

    # if debug.predictions and prediction is not None:
    #     stats = stats._extend(predictions = prediction_stats(encoding, prediction))

    # if debug.boxes:
    #     stats = stats._extend(
    #         boxes=count_classes(encoding.classification, num_classes),
    #     )

        
    return stats


def eval_train(model, encoder, debug = struct(), device=torch.cuda.current_device()):

    def f(data):

        image = data.image.to(device)
        norm_data = normalize_batch(image)
        prediction = model(norm_data)

        loss = encoder.loss(data.encoding, prediction, device=device)

        stats = train_statistics(data, loss, prediction, data.encoding, debug, device)
        return struct(error = loss.total / image.data.size(0), statistics=stats, size = data.image.size(0))

    return f


def summarize_train_stats(name, results, classes, log):
    totals = sum_results(results)
    avg = totals._subset('loss', 'instances', 'error', 'boxes') / totals.size

    log.scalars(name + "/loss", avg.loss._extend(total = avg.error))

    class_names = [c.name.name for c in classes]

    # class_counts = {"class_{}".format(c):count for c, count in zip(class_names, totals.class_instances) }
    # log.scalars(name + "/instances",
    #     struct(total = totals.instances, **class_counts))

    if 'boxes' in totals:
        log_boxes(name, class_names, totals.boxes / totals.size,  log)

    if 'predictions' in totals:
        log_predictions(name, class_names, totals.predictions, log)

    loss_str = " + ".join(["{} : {:.3f}".format(k, v) for k, v in sorted(avg.loss.items())])
    return ('n: {}, instances : {:.2f}, loss: {} = {:.3f}'.format(totals.size, avg.instances, loss_str, avg.error))


def summarize_train(name, results, classes, epoch, log):
    summary = summarize_train_stats(name, results, classes, log)
    print('{} epoch: {} {}'.format(name, epoch, summary))


def axis_splits(size, eval_size, min_overlap=0):
    if eval_size >= size:
        return [(0, size)]

    n = math.ceil((size - min_overlap) / (eval_size - min_overlap))
    overlap = (n * eval_size - size) / (n - 1)
    size_crop = eval_size - overlap
    offsets = [int(i * size_crop) for i in range(n)]
    return [(x, x + eval_size) for x in offsets]

def image_splits(size, eval_size, overlap=0):
    w, h = size
    ex, ey = eval_size
    return [((lx, ly), (ux, uy))
        for lx, ux in axis_splits(w, ex, overlap)
        for ly, uy in axis_splits(h, ey, overlap) ]

def split_image(image, eval_size, overlap=0):
    def sub_image(ranges):
        (lx, ly), (ux, uy) = ranges
        return ((lx, ly), image.narrow(0, ly, uy - ly).narrow(1, lx, ux - lx))

    size = (image.size(1), image.size(0))
    return [ sub_image(r) for r in image_splits(size, eval_size, overlap) ]



def evaluate_image(model, image, encoder, nms_params=box.nms_defaults, crop_boxes=False, device=torch.cuda.current_device()):
    model.eval()
    with torch.no_grad():
        prediction, _ = evaluate_decode(model, image, encoder=encoder, crop_boxes=crop_boxes, device=device)
        return  encoder.nms(prediction, nms_params=nms_params)

def evaluate_raw(model, image, device):
    if image.dim() == 3:
        image = image.unsqueeze(0)

    assert image.dim() == 4, "evaluate: expected image of 4d  [1,H,W,C] or 3d [H,W,C]"
    
    def detach(p):
        return p.detach()[0]

    norm_data = normalize_batch(image).to(device)
    predictions = model(norm_data)._map(detach)

    #gc.collect()
    return predictions

def evaluate_decode(model, image, encoder, device, offset = (0, 0), crop_boxes=False):
    raw = evaluate_raw(model, image, device=device)
    p = encoder.decode(image, raw, crop_boxes=crop_boxes)

    offset = torch.Tensor([*offset, *offset]).to(device)
    return p._extend(bbox = p.bbox + offset), raw

def test_loss(data, encoder, encoding, prediction, debug = struct(), device=torch.cuda.current_device()):
    def unsqueeze(p):
        return p.unsqueeze(0)

    if prediction is not None:
        prediction = prediction._map(unsqueeze)

    loss = encoder.loss(encoding, prediction, device=device)
    return train_statistics(data, loss, prediction, encoding, debug, device)


eval_defaults = struct(
    overlap = 256,
    split = False,

    image_size = (600, 600),
    batch_size = 1,
    nms_params = box.nms_defaults,
    crop_boxes = False,

    device=torch.cuda.current_device(),
    debug = ()
)  

def evaluate_full(model, data, encoder, params=eval_defaults):
    model.eval()
    with torch.no_grad():
        prediction, raw = evaluate_decode(model, data.image.squeeze(0), encoder, 
            device=params.device, crop_boxes=params.crop_boxes)

        train_stats = test_loss(data, encoder, data.encoding, raw, debug=params.debug, device=params.device)
        return encoder.nms(prediction, nms_params=params.nms_params), train_stats


def evaluate_split(model, data, encoder, params=eval_defaults):
    model.eval()
    with torch.no_grad():
        splits = split_image(data.image.squeeze(0), params.image_size, params.overlap)

        outputs = [evaluate_decode(model, image, encoder, device=params.device, offset=offset) for offset, image in splits]
        prediction, raw = zip(*outputs)

        #train_stats = test_loss(data, encoder, data.encoding, prediction, debug=params.debug, device=params.device)
        return encoder.nms(cat_tables(prediction), nms_params=params.nms_params), None


def eval_test(model, encoder, params=eval_defaults):
    evaluate = evaluate_split if params.split else evaluate_full

    def f(data):
        prediction, train_stats = evaluate(model, data, encoder, params)
        return struct (
            id = data.id,
            target = data.target._map(Tensor.to, params.device),

            prediction = prediction,

            # for summary of loss
            instances=data.lengths.sum().item(),
            train_stats = train_stats,

            size = data.image.size(0),
        )
    return f


def percentiles(t, n=100):
    assert t.dim() == 1
    return torch.from_numpy(np.percentile(t.numpy(), np.arange(0, n)))

def mean(xs):
    return sum(xs) / len(xs)


def condense_pr(pr, n=400): 
    positions = [0]
    size = pr.false_positives.size(0)
    i = 0

    for t in range(0, n):
        while pr.recall[i] <= (t / n) and i < size:
            i = i + 1

        if i < size:
            positions.append(i)

    t = torch.LongTensor(positions)

    return struct (
        recall = pr.recall[t],
        precision = pr.precision[t],
        confidence = pr.confidence[t],

        false_positives = pr.false_positives[t],
        false_negatives = pr.false_negatives[t],
        true_positives  = pr.true_positives[t]
    )
    

def compute_thresholds(pr):
    
    f1 = 2 * (pr.precision * pr.recall) / (pr.precision + pr.recall)

    def find_threshold(t):
        diff = pr.false_positives - pr.false_negatives
        p = int((t / 100) * pr.n)

        zeros = (diff + p == 0).nonzero()
        i = 0 if zeros.size(0) == 0 else zeros[0]

        return pr.confidence[i].item()

    margin = 10

    return struct (
        lower   = find_threshold(-margin), 
        middle  = find_threshold(0), 
        upper   = find_threshold(margin)
    )

def threshold_count(confidence, thresholds):
    d = {k : (confidence > t).sum().item() for k, t in thresholds.items()}
    return Struct(d)



def count_target_classes(image_pairs, class_ids):
    labels = torch.cat([i.target.label for i in image_pairs])
    counts = labels.bincount(minlength = len(class_ids))

    return {k : count for k, count in zip(class_ids, counts)}

def compute_AP(results, classes, conf_thresholds=None):

    class_ids = pluck('id', classes)
    iou_thresholds = list(range(30, 100, 5))

    compute_mAP = evaluate.mAP_classes(results, num_classes = len(class_ids))
    info = transpose_structs ([compute_mAP(t / 100) for t in iou_thresholds])

    info.classes = transpose_lists(info.classes)
    assert len(info.classes) == len(class_ids)

    target_counts = count_target_classes(results, class_ids)

    def summariseAP(ap, class_id = None):
        prs = {t : pr for t, pr in zip(iou_thresholds, ap)}
        mAP = {t : pr.mAP for t, pr in prs.items()}

        class_counts = None

        if None not in [conf_thresholds, class_id]:
            class_counts = threshold_count(prs[50].confidence, conf_thresholds[class_id]
                )._extend(truth = target_counts.get(class_id))
            
        return struct(
            mAP = mAP,
            AP = mean([ap for k, ap in mAP.items() if k >= 50]),

            thresholds = compute_thresholds(prs[50]),
            pr50 = condense_pr(prs[50]),
            pr75 = condense_pr(prs[75]),

            class_counts = class_counts
        )

    return struct (
        total   = summariseAP(info.total),
        classes = {id : summariseAP(ap, id) for id, ap in zip(class_ids, info.classes)}
    )



def summarize_test(name, results, classes, epoch, log, thresholds=None):

    class_names = {c.id : c.name.name for c in classes}

    summary = compute_AP(results, classes, thresholds)
    total, class_aps = summary.total, summary.classes

    mAP_strs ='mAP@30: {:.2f}, 50: {:.2f}, 75: {:.2f}'.format(total.mAP[30], total.mAP[50], total.mAP[75])

    train_stats = filter_none(pluck('train_stats', results))

    train_summary = summarize_train_stats(name, train_stats, classes, log) \
        if len(train_stats) > 0 else ''

    print(name + ' epoch: {} AP: {:.2f} mAP@[0.3-0.95]: [{}] {}'.format(epoch, total.AP * 100, mAP_strs, train_summary))

    log.scalars(name, struct(AP = total.AP * 100.0, mAP30 = total.mAP[30] * 100.0, mAP50 = total.mAP[50] * 100.0, mAP75 = total.mAP[75] * 100.0))

    for k, ap in class_aps.items():
        if ap.class_counts is not None:
            log.scalars(name + "/counts/" + class_names[k], ap.class_counts)

        log.scalars(name + "/thresholds/" + class_names[k], ap.thresholds)


    aps = {class_names[k] : ap for k, ap in class_aps.items()}
    aps['total'] = total

    for k, ap in aps.items():
        log.pr_curve(name + "/pr50/" + k, ap.pr50)
        log.pr_curve(name + "/pr75/" + k, ap.pr75)

    if len(classes) > 1:
        log.scalars(name + "/mAP50", {k : ap.mAP[50] * 100.0 for k, ap in aps.items()})
        log.scalars(name + "/mAP75", {k : ap.mAP[75] * 100.0 for k, ap in aps.items()})

        log.scalars(name + "/AP", {k : ap.AP * 100.0 for k, ap in aps.items()})


    return total.AP, {k : ap.thresholds for k, ap in class_aps.items()}
