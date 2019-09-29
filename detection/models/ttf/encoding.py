import sys
import math

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np

from detection import box, display


from tools import struct, table, show_shapes, sum_list, cat_tables


def gaussian_2d(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def clipped_gaussian(image_size, extents, alpha):

    radius = (extents.size / (2. * alpha)).int()
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
    box_target =  areas.new_ones(h, w, 4)
    

    for (label, target_box) in zip(target.classification[boxes_ind], target.bbox[boxes_ind]):
        assert label < num_classes

        extents = box.extents(target_box)
        area = extents.size.dot(extents.size)

        for gaussian, slices in clipped_gaussian(heatmap_size, extents, params.alpha):
            gaussian = gaussian.type_as(heatmap)

            local_heatmap = heatmap[label][slices]
            torch.max(gaussian, local_heatmap, out=local_heatmap)
            
            mask = gaussian > 0
            box_target[slices][mask] = target_box
            box_weight[slices].where(~mask,  gaussian * area.log() / gaussian.sum())  

    return struct(heatmap=heatmap, box_target=box_target, box_weight=box_weight)


def decode_boxes(predictions, centres, scale_factor=16):
    lower, upper = box.split(predictions)

    lower = centres - lower * scale_factor
    upper = centres + upper * scale_factor

    return box.join(lower, upper)


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
        classification = torch.LongTensor(n).random_(0, classes)
    )


if __name__ == "__main__":
    from tools.image import cv

    size = 600
    target = random_target(centre_range=(-20, 620), size_range=(10, 100), n=100)

    layer = 0
    encoded = encode_layer(target, (size, size), 0, 3, struct(alpha=0.54))
    h = encoded.heatmap.permute(1, 2, 0).contiguous()
    
    for b in target.bbox / (2**layer):
        h = display.draw_box(h, b, thickness=1, color=(255, 0, 255, 255))

    cv.display(h)    



