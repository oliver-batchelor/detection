import math

import torch
from tools import struct, Table, show_shapes

from detection import box


def make_boxes(box_sizes, box_dim, device=torch.device('cpu')):
    stride, w, h = box_dim
    n = len(box_sizes)

    sx, sy = stride, stride

    xs = torch.arange(0, w, device=device, dtype=torch.float).add_(0.5).mul_(sx).view(1, w, 1, 1).expand(h, w, n, 1)
    ys = torch.arange(0, h, device=device, dtype=torch.float).add_(0.5).mul_(sy).view(h, 1, 1, 1).expand(h, w, n, 1)

    box_sizes = torch.tensor(box_sizes, device=device, dtype=torch.float).view(1, 1, n, 2).expand(h, w, n, 2)
    boxes = torch.cat([xs, ys, box_sizes], 3).view(-1, 4)

    return boxes


def crop_anchors(boxes, image_dim):
    return box.extents_form(clamp(box.point_form(boxes), (0, 0), image_dim))    

def make_anchors(box_sizes, layer_dims, device=torch.device('cpu')):
    boxes = [make_boxes(boxes, box_dim, device) for boxes, box_dim in zip(box_sizes, layer_dims)]
    return torch.cat(boxes, 0)


def anchor_sizes(size, aspects, scales):
    def anchor(s, ar):
        return (s * math.sqrt(ar), s / math.sqrt(ar))

    return [anchor(size * scale, ar) for scale in scales for ar in aspects]


def encode(target, anchor_boxes, params):
    n = anchor_boxes.size(0)
    m = target.bbox.size(0)

    if m == 0: return struct (
        location        = target.bbox.new_zeros(n, 4), 
        classification  = target.bbox.new_zeros(n, dtype=torch.long)
    )

    ious = box.iou_matrix(box.point_form(anchor_boxes), target.bbox)

    if params.top_anchors > 0:
        top_ious, inds = ious.topk(params.top_anchors, dim = 0)
        ious = ious.scatter(0, inds, top_ious * 2)

    max_ious, max_ids = ious.max(1)

    class_target = encode_classes(target.label, max_ious, max_ids, 
        match_thresholds=params.match_thresholds)


    location = target.bbox[max_ids]
    if params.location_loss == "l1":
        location = encode_boxes(location, anchor_boxes) 

    
    return struct (location  = location, classification = class_target)


def encode_classes(label, max_ious, max_ids, match_thresholds=(0.4, 0.5)):

    match_neg, match_pos = match_thresholds
    assert match_pos >= match_neg

    class_target = 1 + label[max_ids]
    class_target[max_ious <= match_neg] = 0 # negative label is 0

    ignore = (max_ious > match_neg) & (max_ious <= match_pos)  # ignore ious between [0.4,0.5]
    class_target[ignore] = -1  # mark ignored to -1

    return class_target

def encode_boxes(boxes, anchor_boxes):
    '''We obey the Faster RCNN box coder:
        tx = (x - anchor_x) / anchor_w
        ty = (y - anchor_y) / anchor_h
        tw = log(w / anchor_w)
        th = log(h / anchor_h)'''
    boxes_pos, boxes_size = box.split(box.extents_form(boxes))
    anchor_pos, anchor_size = box.split(anchor_boxes)

    loc_pos = (boxes_pos - anchor_pos) / anchor_size
    loc_size = torch.log(boxes_size/anchor_size)
    return torch.cat([loc_pos,loc_size], 1)


def decode(prediction, anchor_boxes):
    '''Decode (encoded) prediction and anchor boxes to give detected boxes.
    Args:
      preditction: (tensor) box prediction in encoded form, sized [n, 4].
      anchor_boxes: (tensor) bounding boxes in extents form, sized [m, 4].
    Returns:
      boxes: (tensor) detected boxes in point form, sized [k, 4].
      label: (tensor) detected class label [k].
    '''
    assert prediction.shape == anchor_boxes.shape

    loc_pos, loc_size = box.split(prediction)
    anchor_pos, anchor_size = box.split(anchor_boxes)

    pos = loc_pos * anchor_size + anchor_pos
    sizes = loc_size.exp() * anchor_size
    
    return box.point_form(torch.cat([pos, sizes], pos.dim() - 1))



def decode_nms(loc_preds, class_preds, anchor_boxes, nms_params):
    assert loc_preds.dim() == 2 and class_preds.dim() == 2

    prediction = decode(loc_preds, class_preds, anchor_boxes)
    return nms(prediction, nms_params).type_as(prediction.label)
