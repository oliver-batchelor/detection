import sys
import math

import torch
import torch.nn as nn
from torch import Tensor

import itertools
import torchvision.models as m

import models.pretrained as pretrained
from detection import box

from models.common import Named, Parallel, image_size

from models.feature_pyramid import feature_pyramid, init_weights, init_classifier, join_output, residual_subnet, pyramid_parameters
from tools import struct, table, show_shapes, sum_list, cat_tables, stack_tables

from tools.parameters import param, choice, parse_args, parse_choice, make_parser, group
from collections import OrderedDict

from . import anchor, loss


class Encoder:
    def __init__(self, start_layer, box_sizes, class_weights, params, device = torch.device('cpu')):
        self.anchor_cache = {}

        self.box_sizes = box_sizes
        self.start_layer = start_layer

        self.class_weights=class_weights
        self.params = params
        self.device = device

    def to(self, device):
        self.device = device
        self.anchor_cache = {k: anchors.to(device) 
            for k, anchors in self.anchor_cache.items()}

    def anchors(self, input_size):
        def layer_size(i):
            stride = 2 ** i
            return (stride, max(1, math.ceil(input_size[0] / stride)), max(1, math.ceil(input_size[1] / stride)))

        input_args = (input_size, self.params.crop_boxes)

        if not (input_args in self.anchor_cache):
            layer_dims = [layer_size(self.start_layer + i) for i in range(0, len(self.box_sizes))]
            anchors = anchor.make_anchors(self.box_sizes, layer_dims, device=self.device)
            if self.params.crop_boxes:
                anchors = anchor.crop_anchors(anchors, input_size)

            self.anchor_cache[input_args] = anchors

        return self.anchor_cache[input_args]

    def encode(self, inputs, target):
        anchor_boxes = self.anchors(image_size(inputs))
        target = anchor.encode(target, anchor_boxes, self.params) 
        
        return struct()

    def decode(self, inputs, prediction):
        assert prediction.location.dim() == 2 and prediction.classification.dim() == 2

        anchor_boxes = self.anchors(image_size(inputs))

        bbox = anchor.decode(prediction.location, anchor_boxes)
        confidence, label = prediction.classification.max(1)

        if self.params.crop_boxes:
            box.clamp(bbox, (0, 0), inputs)

        return table(bbox = bbox, confidence = confidence, label = label)

       
    def loss(self, inputs, target, encoding, prediction):

        anchor_boxes = self.anchors(image_size(inputs))
        target = stack_tables([anchor.encode(t, anchor_boxes, self.params) for t in target])

        class_loss = loss.class_loss(target.classification, prediction.classification,  class_weights=self.class_weights)
        loc_loss = 0

        if self.params.location_loss == "l1":
            loc_loss = loss.l1(target.location, prediction.location, target.classification) 
        elif self.params.location_loss == "giou":
            bbox = anchor.decode(prediction.location, anchor_boxes.unsqueeze(0).expand(prediction.location.size()))
            loc_loss = loss.giou(target.location, bbox, target.classification)

        return struct(classification = class_loss / self.params.balance, location = loc_loss)
 

    def nms(self, prediction, nms_params=box.nms_defaults):
        return box.nms(prediction, nms_params)
    



def check_equal(*elems):
    first, *rest = elems
    return all(first == elem for elem in rest)


def output(features, n, layers, init=init_weights):
    modules = [(str(i), residual_subnet(features, n)) for i in layers]
    for k, module in modules:
        module.apply(init)

    return Parallel(OrderedDict(modules))


class RetinaNet(nn.Module):

    def __init__(self, backbone_name, first, depth, features=32, num_boxes=9, num_classes=2, shared=False):
        super().__init__()

        def named(modules):
            modules = [(str(i + first), module) for i, module in enumerate(modules)]
            return OrderedDict(modules)
       
        self.num_classes = num_classes

        self.outputs = Named(
            location=output(features, 4 * num_boxes, range(first, depth)),
            classification=output(features, num_classes * num_boxes, range(first, depth), init=init_classifier)
        )

        self.pyramid = feature_pyramid(backbone_name=backbone_name, features=features, first=first, depth=depth)     

    def forward(self, input):
        features = self.pyramid(input)
        output = self.outputs(features)

        return struct(
            classification = torch.sigmoid(join_output(output.classification , self.num_classes)),
            location = join_output(output.location, 4)
        )




parameters = struct(
   
    anchor_scale = param (4, help = "anchor scale relative to box stride"),
    shared    = param (False, help = "share weights between network heads at different levels"),
    square    = param (False, help = "restrict box outputs (and anchors) to square"),

    params = group('parameters',
        pos_match = param (0.5, help = "lower iou threshold matching positive anchor boxes in training"),
        neg_match = param (0.4,  help = "upper iou threshold matching negative anchor boxes in training"),

        crop_boxes      = param(False, help='crop boxes to the edge of the image patch in training'),
        top_anchors     = param(1,     help='select n top anchors for ground truth regardless of iou overlap (0 disabled)'),

        location_loss =  param ("l1", help = "location loss function (giou | l1)"),
        balance = param(4., help = "loss = class_loss / balance + location loss")
    ),

    pyramid = group('pyramid_parameters', **pyramid_parameters)
  )



def anchor_sizes(start, depth, anchor_scale=4, square=False, tall=False):

    aspects = [1/2, 1, 2]
    if square:
        aspects = [1]
    elif tall:
        aspects = [1/8, 1/4, 1/2]

    scales = [1, pow(2, 1/3), pow(2, 2/3)]

    return len(aspects) * len(scales), \
        [anchor.anchor_sizes(anchor_scale * (2 ** i), aspects, scales) for i in range(start, depth)]



def create(args, dataset_args):
    num_classes = len(dataset_args.classes)

    num_boxes, box_sizes = anchor_sizes(args.first, args.depth, anchor_scale=args.anchor_scale, square=args.square)

    model = RetinaNet(backbone_name=args.backbone, first=args.first, depth=args.depth, num_boxes=num_boxes, \
        num_classes=num_classes, features=args.features, shared=args.shared)

    assert args.location_loss in ["l1", "giou"]
    params = struct(
        crop_boxes=args.crop_boxes, 
        match_thresholds=(args.neg_match, args.pos_match), 
        top_anchors = args.top_anchors,
        location_loss =  args.location_loss,
        balance = args.balance
    )

    class_weights = [c.name.get('weighting', 0.25) for c in dataset_args.classes]
    encoder = Encoder(args.first, box_sizes, class_weights=class_weights, params=params)

    return model, encoder
    

model = struct(create=create, parameters=parameters)

if __name__ == '__main__':

    _, *cmd_args = sys.argv

    parser = make_parser('object detection', model.parameters)
    model_args = struct(**parser.parse_args().__dict__)

    classes = [
        struct(name=struct(weighting=0.25)),
        struct(name=struct(weighting=0.25))
    ]

    model, encoder = model.create(model_args, struct(classes = classes, input_channels = 3))

    x = torch.FloatTensor(4, 3, 370, 500)
    out = model.cuda()(x.cuda())

    print(show_shapes(out))
