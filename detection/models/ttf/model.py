import sys
import math

import torch
import torch.nn as nn
from torch import Tensor

import itertools
import torchvision.models as m

import torch.nn.functional as F

import models.pretrained as pretrained
from detection import box, detection_table

from tools.image.transforms import normalize_batch
from models.common import Named, Parallel, image_size

from models.feature_pyramid import feature_map, init_weights, init_classifier, \
    join_output, residual_subnet, pyramid_parameters

from tools import struct, table, shape, sum_list, cat_tables, stack_tables, tensors_to

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
        return self

    def _centres(self, w, h):

        self.centre_map = encoding.expand_centres(self.centre_map, (w, h), device=self.device)
        return self.centre_map[:h, :w]

    def encode(self, inputs, target):
        input_size = image_size(inputs)
        num_classes = len(self.class_weights)
        return encoding.encode_layer(target, input_size, self.layer, num_classes, self.params) 


    def decode(self, inputs, prediction, nms_params=detection_table.nms_defaults):
        (classification, location) = prediction

        h, w, _ = classification.shape
        boxes = encoding.decode_boxes(self._centres(w, h), location, self.stride)
        return encoding.decode(classification, boxes, nms_params=nms_params)

    @property
    def debug_keys(self):
        return ["heatmap", "maxima", "heatmap_target", "target_weight"]
    
    def debug(self, image, target, prediction, classes):
        (classification, location) = prediction

        h, w, num_classes = classification.shape
        input_size = image_size(image)

        class_colours = [c.colour for c in classes]
        encoded_target = encoding.encode_layer(target, input_size, self.layer, num_classes, self.params)

        return struct(
            heatmap=encoding.show_heatmap(classification, class_colours), 
            maxima=encoding.show_local_maxima(classification),
            heatmap_target=encoding.show_heatmap(encoded_target.heatmap, class_colours),
            target_weight=encoding.show_weights(encoded_target.box_weight, (1, 0, 0))
        )

       
    def loss(self, inputs, target, encoded_target, prediction):
        (classification, location) = prediction
        batch, h, w, num_classes = classification.shape
          
        class_loss = loss.class_loss(encoded_target.heatmap, classification,  class_weights=self.class_weights)
        centres = self._centres(w, h).unsqueeze(0).expand(batch, h, w, -1)
                     
        box_prediction = encoding.decode_boxes(centres, location, 1)
        loc_loss = loss.giou(encoded_target.box_target, box_prediction, encoded_target.box_weight)

        return struct(classification = class_loss / self.params.balance, location = loc_loss)
    


class TTFNet(nn.Module):

    def __init__(self, pyramid, features=32, num_classes=2, head_blocks=2, scale_factor=4):
        super().__init__()

        self.num_classes = num_classes
        self.scale_factor = scale_factor

        self.regressor = residual_subnet(features, 4, num_blocks=head_blocks)
        self.classifier = residual_subnet(features, num_classes, num_blocks=head_blocks)

        self.classifier.apply(init_classifier)
        self.regressor.apply(init_weights)

        self.pyramid = pyramid 

    def forward(self, input):
        permute = lambda layer: layer.permute(0, 2, 3, 1).contiguous()
        features = self.pyramid(input)
        
        return (
            permute(self.classifier(features).sigmoid()),
            (permute(self.regressor(features)) * self.scale_factor).clamp_(min=0)
         )


pyramid_parameters = pyramid_parameters

parameters = struct(
  
    params = group('parameters',
        alpha   = param(0.54, help = "control size of heatmap gaussian sigma = alpha * length / 6"),
        balance = param(1., help = "loss = class_loss / balance + location loss")
    ),

    pyramid = group('pyramid_parameters', **pyramid_parameters._extend (
        first     = param (2, help = "first layer of feature maps, scale = 1 / 2^first"),
    )),

    head_blocks = param (2, help = "number of residual blocks in network heads"),
  )


def create(args, dataset_args):
    num_classes = len(dataset_args.classes)

    feature_gen = feature_map(backbone_name=args.backbone, first=args.first, depth=args.depth, features=args.features, decode_blocks=args.decode_blocks)     
    model = TTFNet(feature_gen, features=args.features, num_classes=num_classes, head_blocks=args.head_blocks)

    params = struct(
        alpha=args.alpha, 
        balance = args.balance
    )

    class_weights = [c.get('weighting', 0.25) for c in dataset_args.classes]
    encoder = Encoder(args.first, class_weights=class_weights, params=params)

    return model, encoder
    

model = struct(create=create, parameters=parameters)

if __name__ == '__main__':

    _, *cmd_args = sys.argv

    parser = make_parser('object detection', model.parameters)
    model_args = struct(**parser.parse_args().__dict__)

    classes = [
        struct(weighting=0.25),
        struct(weighting=0.25)
    ]

    model, encoder = model.create(model_args, struct(classes = classes, input_channels = 3))

    device = device = torch.cuda.current_device()

    model.to(device)
    encoder.to(device)

    x = torch.FloatTensor(4, 370, 500, 3).uniform_(0, 255)
    out = model.cuda()(normalize_batch(x).cuda())
    print(shape(out))

    target = encoding.random_target(classes=len(classes))
    target = tensors_to(target, device='cuda:0')

    loss = encoder.loss(x, [target, target, target, target], struct(), out)
    print(loss)

    


