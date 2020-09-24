# Originally by Alexander (Max) deGroot
# https://github.com/amdegroot/ssd.pytorch.git


from tools import struct, Table, shape
import torch

import math
import torchvision.ops as torchvision


def split(boxes):
    return boxes[..., :2],  boxes[..., 2:]

def split4(boxes):
    return boxes[..., 0],  boxes[..., 1], boxes[..., 2], boxes[..., 3]


def join(lower, upper):
    assert lower.shape == upper.shape
    return torch.cat([lower, upper], lower.dim() - 1)


def extents(boxes):
    lower, upper = split(boxes)
    return struct(centre = (lower + upper) * 0.5, size = upper - lower)

def extents_form(boxes):
    b = extents(boxes)
    return torch.cat([b.centre, b.size], boxes.dim() - 1)

def point_form(boxes):
    centre, size = split(boxes)
    radius = size * 0.5
    return torch.cat([centre - radius, centre + radius], boxes.dim() - 1)



def transform(boxes, offset=(0, 0), scale=(1, 1)):
    lower, upper = boxes[:, :2], boxes[:, 2:]

    offset, scale = torch.Tensor(offset), torch.Tensor(scale)

    lower = lower.add(offset).mul(scale)
    upper = upper.add(offset).mul(scale)

    return torch.cat([lower.min(upper), lower.max(upper)], 1)


def transpose(boxes):
    x1, y1, x2, y2 = split4(boxes)
    return torch.stack([y1, x1, y2, x2], boxes.dim() - 1)


def flip_horizontal(boxes, width):
    x1, y1, x2, y2 = split4(boxes)
    return torch.stack([width - x2, y1, width - x1, y2], boxes.dim() - 1)

def flip_vertical(boxes, height):
    x1, y1, x2, y2 = split4(boxes)
    return torch.stack([x1, height - y2, x2, height - y1], boxes.dim() - 1)



def filter_invalid(target):
    boxes = target.bbox

    valid = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
    return target[valid.nonzero(as_tuple=False).squeeze(1)]

def filter_hidden(target, lower, upper, min_visible=0.0):
    bounds = torch.Tensor([[*lower, *upper]])
    overlaps = (intersect_matrix(bounds, target.bbox) / area(target.bbox)).squeeze(0)
    return target._index_select(overlaps.gt(min_visible).nonzero(as_tuple=False).squeeze(1))



def area(boxes):
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    return (x2-x1) * (y2-y1)

def clamp(boxes, lower, upper):

    boxes[:, 0].clamp_(min = lower[0])
    boxes[:, 1].clamp_(min = lower[1])
    boxes[:, 2].clamp_(max = upper[0])
    boxes[:, 3].clamp_(max = upper[1])

    return boxes


def intersect_matrix(box_a, box_b):
    """ Intersection matrix of bounding boxes
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,4].
      box_b: (tensor) bounding boxes, Shape: [m,4].
    Return:
      (tensor) intersection area, Shape: [n,m].
    """
    n = box_a.size(0)
    m = box_b.size(0)


    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(n, m, 2),
                       box_b[:, 2:].unsqueeze(0).expand(n, m, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(n, m, 2),
                       box_b[:, :2].unsqueeze(0).expand(n, m, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def intersect(box_a, box_b):
    assert box_a.shape == box_b.shape

    max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
    min_xy = torch.max(box_a[:, :2], box_b[:, :2])

    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]

def union_matrix(box_a, box_b):
    """Compute the union area matrix between two sets of boxes in point form.
    Args:
        box_a, box b: Bounding boxes in point form. shapes ([n, 4], [m, 4])
    Return:
        intersection: (tensor) Shape: [n, m]
        union: (tensor) Shape: [n, m]
    """
    inter = intersect_matrix(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [n,m]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [n,m]
    unions = area_a + area_b - inter
    return inter, unions  # [n,m]    

def iou_matrix(box_a, box_b):
    """Compute the IOU of two sets of boxes in point form.
    Args:
        box_a, box b: Bounding boxes in point form. shapes ([n, 4], [m, 4])
    Return:
        jaccard overlap: (tensor) Shape: [n, m]
    """
    inter, union = union_matrix(box_a, box_b)
    return inter / union

def union(box_a, box_b):  
    assert box_a.shape == box_b.shape

    inter = intersect(box_a, box_b)

    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1]))
    unions = area_a + area_b - inter
    return inter, unions 

def iou(box_a, box_b):  
    assert box_a.shape == box_b.shape

    inter, unions = union(box_a, box_b)
    return conditional_div(inter, unions)

def merge(box_a, box_b):
    l1, u1 = split(box_a)
    l2, u2 = split(box_b)

    return torch.cat([l1.min(l2), u1.max(u2)], 1)

def conditional_div(a, b):
    return a / torch.where(b == 0, b.new_ones(b.shape), b)

def giou(box_a, box_b):
    hull = area(merge(box_a, box_b))

    inter, unions = union(box_a, box_b)

    iou =  conditional_div(inter, unions)
    giou = conditional_div(hull - unions, hull)

    return iou - giou




def random_points(r, n):
    lower, upper = r
    return torch.FloatTensor(n, 2).uniform_(*r)

def random(centre_range, size_range, n):
    centre = random_points(centre_range, n)
    extents = random_points(size_range, n) * 0.5

    return torch.cat([centre - extents, centre + extents], 1)


    

if __name__ == "__main__":

    boxes1 = random((0, 10), (50, 100), 10)
    boxes2 = random((0, 10), (50, 100), 10)
    
    print(iou(boxes1, boxes2), giou(boxes1, boxes2))

