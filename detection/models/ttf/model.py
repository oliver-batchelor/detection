import sys
import math

import torch
import torch.nn as nn
from torch import Tensor

import itertools
import torchvision.models as m

import models.pretrained as pretrained
from detection import box

from tools.image.transforms import normalize_batch
from models.common import Named, Parallel, image_size

from models.feature_pyramid import feature_map, init_weights, init_classifier, \
    join_output, residual_subnet, pyramid_parameters

from tools import struct, table, show_shapes, sum_list, cat_tables, stack_tables, tensors_to

from tools.parameters import param, choice, parse_args, parse_choice, make_parser, group
from collections import OrderedDict

from . import encoding, loss




class Encoder:
    def __init__(self, layer, class_weights, params, device = torch.device('cpu')):
        self.centre_map = torch.FloatTensor(0, 0, 2).to(device)

        self.layer = layer
        self.stride =  stride = 2 ** layer
        self.class_weights = class_weights

        self.params = params
        self.device = device

    def to(self, device):
        self.device = device
        self.centre_map = self.centre_map.to(device)

    def _centres(self, w, h):
        self.centre_map = encoding.expand_centres(self.centre_map, self.stride, (w, h), device=self.device)
        return self.centre_map[:h, :w]

    def encode(self, inputs, target):
        return struct()

    def decode(self, inputs, prediction, nms_params=box.nms_defaults):
        h, w, _ = prediction.classification.shape
        
        centres = self._centres(w, h)
        boxes = decode_boxes(prediction.location, centres)

        return encoding.decode(prediction.classification, boxes, nms_params=nms_params)

       
    def loss(self, inputs, target, enc, prediction):
        batch, h, w, num_classes = prediction.classification.shape
        input_size = image_size(inputs)

        targets = [encoding.encode_layer(t, input_size, self.layer, num_classes, self.params) for t in target]
        target = stack_tables(targets)

        class_loss = loss.class_loss(target.heatmap, prediction.classification,  class_weights=self.class_weights)

        centres = self._centres(w, h).unsqueeze(0).expand(batch, h, w, -1)
        box_prediction = encoding.decode_boxes(prediction.location, centres)

        loc_loss = loss.giou(target.box_target, box_prediction, target.box_weight)

        return struct(classification = class_loss / self.params.balance, location = loc_loss)
    


class TTFNet(nn.Module):

    def __init__(self, backbone_name, first, depth, features=32, num_classes=2, scale_factor=16):
        super().__init__()

        self.num_classes = num_classes
        self.scale_factor = scale_factor

        self.regressor = residual_subnet(features, 4)
        self.classifier = residual_subnet(features, num_classes)

        self.classifier.apply(init_classifier)
        self.regressor.apply(init_weights)

        self.pyramid = feature_map(backbone_name=backbone_name, features=features, first=first, depth=depth)     

    def forward(self, input):
        permute = lambda layer: layer.permute(0, 2, 3, 1).contiguous()
        features = self.pyramid(input)
        
        return struct(
            location = permute(self.regressor(features)) * self.scale_factor,
            classification = permute(self.classifier(features).sigmoid())
         )


parameters = struct(
    anchor_scale = param (4, help = "anchor scale relative to box stride"),
    
    params = group('parameters',
        alpha   = param(1., help = "control size of heatmap gaussian sigma = length / (6 * alpha)"),
        balance = param(4., help = "loss = class_loss / balance + location loss")
    ),

    pyramid = group('pyramid_parameters', **pyramid_parameters)
  )



def create(args, dataset_args):
    num_classes = len(dataset_args.classes)

    model = TTFNet(backbone_name=args.backbone, first=args.first, depth=args.depth,
        num_classes=num_classes, features=args.features)

    params = struct(
        alpha=args.alpha, 
        balance = args.balance
    )

    class_weights = [c.name.get('weighting', 0.25) for c in dataset_args.classes]
    encoder = Encoder(args.first, class_weights=class_weights, params=params)

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

    device = device = torch.cuda.current_device()

    model.to(device)
    encoder.to(device)

    x = torch.FloatTensor(4, 370, 500, 3).uniform_(0, 255)
    out = model.cuda()(normalize_batch(x).cuda())
    print(show_shapes(out))

    target = encoding.random_target(classes=len(classes))
    target = tensors_to(target, device='cuda:0')

    loss = encoder.loss(x, [target, target, target, target], struct(), out)
    print(loss)

    


