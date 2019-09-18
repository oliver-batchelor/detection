import sys
import math

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np

from detection import box

from tools import struct, table, show_shapes, sum_list, cat_tables


def gaussian_2d(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_truncate_gaussian(heatmap, center, dim, k=1):
    w, h = dim

    gaussian = heatmap.new_tensor(gaussian_2d((h, w), sigma_x=w / 6, sigma_y=h / 6))
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, w_radius), min(width - x, w_radius + 1)
    top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                        w_radius - left:w_radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


default_match_params = struct(
    alpha = 0.54
)


def layer_size(input_size, i):
    stride = 2 ** i
    return (stride, max(1, math.ceil(input_size[0] / stride)), max(1, math.ceil(input_size[1] / stride)))

def heatmap_layer(target, input_size, layer,  num_classes, match_params=default_match_params):
    stride, heatmap_size = layer_size(input_size, layer)
    return heatmap(target._extend(bbox = target.bbox * (1. / stride)), heatmap_size, num_classes)


def heatmap(target, heatmap_size, num_classes, match_params=default_match_params):

    m = target.bbox.size(0)
    w, h = heatmap_size

    # sort by area, largest boxes first (and least priority)
    box_areas = box.area(target.bbox)
    box_areas, boxes_ind = torch.sort(box_areas, descending=True)

    labels = target.classification[boxes_ind]
    extents = box.extents_form(target.bbox)[boxes_ind]

    centres, size = box.split(extents)
    
    radius = size / 2.
    centers = centres.int()

    
    heatmap = torch.FloatTensor(num_classes, h, w)
    for (l, c, r) in zip(labels, centres, radius):
        assert l < heatmap.size(0)

        draw_truncate_gaussian(heatmap[l], c, r)

    return heatmap


def random_points(r, n):
    lower, upper = r
    return torch.FloatTensor(n, 2).uniform_(*r)


def random_boxes(centre_range, size_range, n):
    centre = random_points(centre_range, n)
    extents = random_points(size_range, n) * 0.5

    return torch.cat([centre - extents, centre + extents], 1)




if __name__ == "__main__":
    from tools.image import cv

    target = struct (
        bbox = random_boxes((0, 600), (20, 100), 20),
        classification = torch.LongTensor(20).random_(0, 3)
    )

    h = heatmap(target, (600, 600), 3)

    cv.display(h)    



