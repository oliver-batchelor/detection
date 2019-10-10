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
from tools import struct, tensor, shape, cat_tables, shape_info, \
    Histogram, ZipList, transpose_structs, transpose_lists, pluck, Struct, filter_none, split_table, tensors_to

from detection import box, evaluate, detection_table
from functools import reduce

import operator


def mean_results(results):
    total = reduce(operator.add, results)
    return total / total.size, total

def sum_results(results):
    return reduce(operator.add, results)

# TODO: move this entirely to the individual object detector
def make_statistics(data, encoder, loss, prediction):

    stats = struct(error=sum(loss.values()),
        loss = loss._map(Tensor.item),
        size = data.image.size(0),
        instances=data.lengths.sum().item(),
    )

    return stats



def eval_train(model, encoder, debug = struct(), device=torch.cuda.current_device()):
    def f(data):
        image = data.image.to(device)
        norm_data = normalize_batch(image)
        prediction = model(norm_data)

        target_table = tensors_to(data.target, device=device)
        encoding = tensors_to(data.encoding, device=device)

        targets = split_table(target_table, data.lengths.tolist())
        loss = encoder.loss(image, targets, encoding, prediction)

        statistics = make_statistics(data, encoder, loss, prediction)
        return struct(error = sum(loss.values()) / image.data.size(0), statistics=statistics, size = data.image.size(0))
    return f


def summarize_train_stats(name, results, classes, log):
    totals = sum_results(results)
    avg = totals._subset('loss', 'instances', 'error') / totals.size

    log.scalars(name + "/loss", avg.loss._extend(total = avg.error))

    loss_str = " + ".join(["{} : {:.3f}".format(k, v) for k, v in sorted(avg.loss.items())])
    return ('n: {}, instances : {:.2f}, loss: {} = {:.3f}'.format(totals.size, avg.instances, loss_str, avg.error))


def summarize_train(name, results, classes, epoch, log):
    summary = summarize_train_stats(name, results, classes, log)
    print('{} epoch: {} {}'.format(name, epoch, summary))



def evaluate_image(model, image, encoder, nms_params=detection_table.nms_defaults,  device=torch.cuda.current_device()):
    model.eval()
    with torch.no_grad():
        batch = image.unsqueeze(0) if image.dim() == 3 else image          
        assert batch.dim() == 4, "evaluate: expected image of 4d  [1,H,W,C] or 3d [H,W,C]"

        norm_data = normalize_batch(batch).to(device)
        prediction = model(norm_data)._map(lambda p: p.detach()[0])

        return struct(detections = encoder.decode(batch, prediction), prediction = prediction)

  
eval_defaults = struct(
    overlap = 256,
    split = False,

    image_size = (600, 600),
    batch_size = 1,
    nms_params = detection_table.nms_defaults,

    device=torch.cuda.current_device(),
    debug = ()
)  

def evaluate_full(model, data, encoder, params=eval_defaults):
    model.eval()
    with torch.no_grad():
        result = evaluate_image(model, data.image, encoder, device=params.device, nms_params=params.nms_params)

        prediction = result.prediction._map(lambda p: p.unsqueeze(0))
        target = tensors_to(data.target, device=params.device)
        encoding = tensors_to(data.encoding, device=params.device)

        loss = encoder.loss(data.image, [target], encoding, prediction)
        statistics = make_statistics(data, encoder, loss, result.prediction)

        return result._extend(statistics=statistics)



def eval_test(model, encoder, params=eval_defaults):
    def f(data):
        result = evaluate_full(model, data, encoder, params)
        return struct (
            id = data.id,
            target = data.target._map(Tensor.to, params.device),

            detections = result.detections,

            # for summary of loss
            instances=data.lengths.sum().item(),
            statistics = result.statistics,

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

    class_names = {c.id : c.name for c in classes}

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
