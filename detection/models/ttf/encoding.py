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

def truncate_gaussian(heatmap, center, w_radius, h_radius, k=1):
    w = w_radius * 2 + 1
    h = h_radius * 2 + 1

    gaussian = heatmap.new_tensor(gaussian_2d((int(h), int(w)), sigma_x=w / 6, sigma_y=h / 6))
    x, y = center

    height, width = heatmap.shape[0:2]

    left, right = min(x, w_radius), min(width - x, w_radius + 1)
    top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

    heatmap[y - top:y + bottom, x - left:x + right] = \
        gaussian[h_radius - top:h_radius + bottom, w_radius - left:w_radius + right]

    return heatmap
   

default_match_params = struct(
    alpha = 0.54
)


def layer_size(input_size, i):
    stride = 2 ** i
    return (stride, max(1, math.ceil(input_size[0] / stride)), max(1, math.ceil(input_size[1] / stride)))

def encode_layer(target, input_size, layer,  num_classes, match_params=default_match_params):
    stride, heatmap_size = layer_size(input_size, layer)
    return encode_target(target._extend(bbox = target.bbox * (1. / stride)), heatmap_size, num_classes)


def encode_target(target, heatmap_size, num_classes, match_params=default_match_params):

    m = target.bbox.size(0)
    w, h = heatmap_size


    # sort by area, largest boxes first (and least priority)
    areas = box.area(target.bbox)
    areas, boxes_ind = torch.sort(areas, descending=True)

    heatmap = areas.new_zeros(num_classes, h, w)
    bbox = struct(
        weight =  areas.new_zeros(h, w),
        target =  areas.new_ones(h, w, 4)
    )

    for (label, target_box) in zip(target.classification[boxes_ind], target.bbox[boxes_ind]):
        assert label < num_classes

        extents = box.extents(target_box)
        area = extents.size.dot(extents.size)
        radius = (extents.size / 2.).int()

        box_heatmap = heatmap.new_zeros(h, w)
        truncate_gaussian(box_heatmap, extents.centre.int(), radius[0].item(), radius[1].item())

        target_inds = box_heatmap > 0 
        local_heatmap = box_heatmap[target_inds]

        heatmap[label] = torch.max(box_heatmap, heatmap[label])

        bbox.target[target_inds] = target_box
        bbox.weight[target_inds] = local_heatmap * area.log() / local_heatmap.sum()    

    return struct(heatmap=heatmap, bbox=bbox)


def random_points(r, n):
    lower, upper = r
    return torch.FloatTensor(n, 2).uniform_(*r)


def random_boxes(centre_range, size_range, n):
    centre = random_points(centre_range, n)
    extents = random_points(size_range, n) * 0.5

    return torch.cat([centre - extents, centre + extents], 1)



if __name__ == "__main__":
    from tools.image import cv

    size = 600

    target = struct (
        bbox = random_boxes((0, size), (50, 200), 20),
        classification = torch.LongTensor(20).random_(0, 3)
    )

    encoded = encode_target(target, (size, size), 3)
    h = encoded.heatmap.permute(1, 2, 0).contiguous()
    
    for b in target.bbox:
        h = display.draw_box(h, b, thickness=1, color=(255, 0, 255, 255))

    cv.display(h)    



