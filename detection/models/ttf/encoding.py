import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np

from detection import box, display
from tools import struct, table, show_shapes, sum_list, cat_tables, show_shapes


def make_centres(w, h, stride, device):               
    x = torch.arange(0, w, device=device, dtype=torch.float).add_(0.5).mul_(stride)
    y = torch.arange(0, h, device=device, dtype=torch.float).add_(0.5).mul_(stride)

    return torch.stack(torch.meshgrid(y, x), dim=2)

def expand_centres(centres, stride, input_size, device):
    w, h = max(1, math.ceil(input_size[0])), max(1, math.ceil(input_size[1]))
    ch, cw, _ = centres.shape

    if ch < h or cw < w:
        return make_centres(max(w, cw), max(h, ch), stride, device=device)
    else:
        return centres


def decode_boxes(predictions, centres):
    lower, upper = box.split(predictions)

    lower = centres - lower
    upper = centres + upper

    return box.join(lower, upper)

def decode(classification, boxes, kernel=3, nms_params=box.nms_defaults):
    h, w, num_classes = classification.shape
    classification = classification.permute(2, 0, 1).contiguous()

    maxima = F.max_pool2d(classification, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
    mask = (maxima == classification) & (maxima >= nms_params.threshold)
    
    maxima.masked_fill_(~mask, 0.)
    confidence, inds = maxima.view(-1).topk(k = min(nms_params.detections, mask.sum()), dim=0)

    labels   = inds // (h * w)
    box_inds = inds % (h * w)
    
    return struct(label = labels, bbox = boxes.view(-1, 4)[box_inds], confidence=confidence)




def gaussian_2d(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def clipped_gaussian(image_size, extents, alpha):

    radius = ((extents.size / 2.) * alpha).int()
    w, h = (radius * 2 + 1).tolist()
    rw, rh = radius.tolist()

    x, y = extents.centre.int().tolist()     
    gaussian = torch.FloatTensor(gaussian_2d((h, w), sigma_x=w / 6, sigma_y=h / 6))

    left, right = min(x, rw), min(image_size[0] - x, rw + 1)
    top, bottom = min(y, rh), min(image_size[1] - y, rh + 1)
    
    if x + right > x - left and y + bottom > y - top:
    
        slices = [slice(y - top, y + bottom), slice(x - left, x + right)]
        clipped = gaussian[rh - top:rh + bottom, rw - left:rw + right]

        return [(clipped, slices)]
    return []


def layer_size(input_size, i):
    stride = 2 ** i
    return stride, (max(1, math.ceil(input_size[0] / stride)), max(1, math.ceil(input_size[1] / stride)))

def encode_layer(target, input_size, layer,  num_classes, params):
    stride, heatmap_size = layer_size(input_size, layer)
    return encode_target(target._extend(bbox = target.bbox * (1. / stride)), heatmap_size, num_classes, params)


def encode_target(target, heatmap_size, num_classes, params):

    m = target.bbox.size(0)
    w, h = heatmap_size

    # sort by area, largest boxes first (and least priority)
    areas = box.area(target.bbox)
    areas, boxes_ind = torch.sort(areas, descending=True)

    heatmap = areas.new_zeros(num_classes, h, w)
    box_weight =  areas.new_zeros(h, w)
    box_target =  areas.new_zeros(h, w, 4)
    
    for (label, target_box) in zip(target.label[boxes_ind], target.bbox[boxes_ind]):
        assert label < num_classes

        extents = box.extents(target_box)
        area = extents.size.dot(extents.size)

        for gaussian, slices in clipped_gaussian(heatmap_size, extents, params.alpha):
            gaussian = gaussian.type_as(heatmap) 

            local_heatmap = heatmap[label][slices]
            torch.max(gaussian, local_heatmap, out=local_heatmap)
            
            loc_weight = gaussian * (area.log() / gaussian.sum())

            mask = loc_weight > box_weight[slices]
            box_target[slices][mask] = target_box
            box_weight[slices][mask] = loc_weight[mask]

    return struct(heatmap=heatmap.permute(1, 2, 0), box_target=box_target, box_weight=box_weight)





def random_points(r, n):
    lower, upper = r
    return torch.FloatTensor(n, 2).uniform_(*r)


def random_boxes(centre_range, size_range, n):
    centre = random_points(centre_range, n)
    extents = random_points(size_range, n) * 0.5

    return torch.cat([centre - extents, centre + extents], 1)



def random_target(centre_range=(0, 600), size_range=(50, 200), classes=3, n=20):
    return struct (
        bbox = random_boxes(centre_range, size_range, n),
        label = torch.LongTensor(n).random_(0, classes)
    )


def show_targets(encoded, target, layer=0):

    h = encoded.heatmap.contiguous()
    w = encoded.box_weight.contiguous() * 255

    for b, l in zip(target.bbox / (2**layer), target.label):

        print(b, l)
        color = [0.2, 0.2, 0.2, 1]
        color[l] = 1

        h = display.draw_box(h, b, thickness=1, color=color)
        w = display.draw_box(w, b, thickness=1, color=color)

    cv.display(torch.cat([h, w.unsqueeze(2).expand_as(h)], dim=1))    

if __name__ == "__main__":
    from tools.image import cv

    size = 600
    target = random_target(centre_range=(0, 600), size_range=(10, 100), n=100)

    layer = 0
    encoded = encode_layer(target, (size, size), layer, 3, struct(alpha=0.54))

    decoded = decode(encoded.heatmap, encoded.box_target)
    print(show_shapes(decoded))
    show_targets(encoded, decoded)

