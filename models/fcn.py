import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import itertools
import torchvision.models as m

import models.pretrained as pretrained
from detection import box

from models.common import Conv, Cascade, UpCascade, Residual, Parallel, Shared,  \
            DecodeAdd, Decode,  basic_block, se_block, reduce_features, replace_batchnorms, identity, GlobalSE

import torch.nn.init as init
from tools import struct, table

from tools.parameters import param, choice, parse_args, parse_choice, make_parser
from collections import OrderedDict


def image_size(inputs):

    if torch.is_tensor(inputs):
        assert(inputs.dim() == 3)
        inputs = inputs.size(1), inputs.size(0)

    assert (len(inputs) == 2)
    return inputs



class Encoder:
    def __init__(self, start_layer, box_sizes):
        self.anchor_cache = {}

        self.box_sizes = box_sizes
        self.start_layer = start_layer


    def anchors(self, input_size, crop_boxes=False):
        def layer_size(i):
            stride = 2 ** i
            return (max(1, math.ceil(input_size[0] / stride)), max(1, math.ceil(input_size[1] / stride)))

        input_args = (input_size, crop_boxes)

        if not (input_args in self.anchor_cache):
            layer_dims = [layer_size(self.start_layer + i) for i in range(0, len(self.box_sizes))]
            self.anchor_cache[input_args] = box.make_anchors(self.box_sizes, layer_dims, input_size, crop_boxes=crop_boxes)

        return self.anchor_cache[input_args]


    def encode(self, inputs, target, crop_boxes=False, match_thresholds=(0.4, 0.5), match_nearest = 0):
        inputs = image_size(inputs)

        return box.encode(target, self.anchors(inputs, crop_boxes), match_thresholds, match_nearest)


    def decode(self, inputs, prediction):
        assert prediction.location.dim() == 2 and prediction.classification.dim() == 2

        inputs = image_size(inputs)
        anchor_boxes = self.anchors(inputs).type_as(prediction.location)

        confidence, label = prediction.classification.max(1)
        return table(bbox = box.decode(prediction.location, anchor_boxes), confidence = confidence, label = label)
        
    def nms(self, prediction, nms_params=box.nms_defaults):
        return box.nms(prediction, nms_params)



def init_weights(m):
    if isinstance(m, nn.Conv2d):
        init.normal_(m.weight, std=0.01)


def init_classifier(m):
    if isinstance(m, nn.Conv2d):
        init.normal_(m.weight, std=0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            prior_prob = 0.001
            b = -math.log((1 - prior_prob)/prior_prob)
            init.constant_(m.bias, b)
            # init.constant_(m.bias, 0)

def check_equal(*elems):
    first, *rest = elems
    return all(first == elem for elem in rest)


class FCN(nn.Module):

    def __init__(self, backbone, box_sizes, layer_names, fine_tune = [], features=32, num_classes=2, shared=False, square=False):
        super().__init__()
       
        self.box_sizes = box_sizes
        self.num_classes = num_classes
        self.square = square

        assert check_equal(len(layer_names), len(box_sizes)), "FCN: layers and box sizes differ in length"

        def named(modules):
            assert len(modules) == len(layer_names)
            return OrderedDict(zip(layer_names, modules))

        self.backbone = backbone

        def make_reducer(size):
            return Conv(size, features, 1)

        def make_decoder():
            decoder = nn.Sequential (
                Residual(basic_block(features, features)),
                Residual(basic_block(features, features))
            )
            # decoder = identity
            return Decode(features, module=decoder)


        encoded_sizes = pretrained.encoder_sizes(self.backbone)
        self.reduce = Parallel(named([make_reducer(size) for size in encoded_sizes]))

        self.decoder = UpCascade(named([make_decoder() for i in encoded_sizes]))


        def output(n):
            return nn.Sequential(
                Residual(basic_block(features, features)),
                Residual(basic_block(features, features)),
                Conv(features, n, 1, bias=True))


        if shared:
            num_boxes = len(box_sizes[0])
            assert all(len(boxes) == num_boxes for boxes in box_sizes), "FCN: for shared heads, number of boxes must be equal on all layers"
            self.classifiers = Shared(output(num_boxes * self.num_classes))
        else:
            self.classifiers = Parallel(named([output(len(boxes) * self.num_classes) for boxes in self.box_sizes]))

        self.localisers = Parallel(named([output(len(boxes) * (3 if square else 4)) for boxes in self.box_sizes]))


        self.new_modules = [self.localisers, self.classifiers, self.reduce, self.decoder]
        nn.ModuleList(self.new_modules).apply(init_weights)
        self.classifiers.apply(init_classifier)

        self.fine_tune = fine_tune


    def forward(self, input):
        layers = self.backbone(input)

  
        layers = self.decoder(self.reduce(layers))
        def join(layers, n):

            def permute(layer):
              out = layer.permute(0, 2, 3, 1).contiguous()
              return out.view(out.size(0), -1, n)

            return torch.cat(list(map(permute, layers)), 1)

        conf = torch.sigmoid(join(self.classifiers(layers), self.num_classes))
        locs = join(self.localisers(layers), 3 if self.square else 4)


        if self.square:
            locs = torch.cat([locs, locs.narrow(2, 2, 1)], dim=2)

        return struct( location = locs, classification = conf )

    def parameter_groups(self, lr, fine_tuning=0.1):
        
        return [
            {'params': self.backbone.parameters(), 'lr':lr, 'modifier': fine_tuning},
            {'params': nn.ModuleList(self.new_modules).parameters(), 'lr':lr, 'modifier': 1.0}
        ]

        # fine_tune = {p:True for m in self.fine_tune for p in  m.parameters()}
        # normal = [p for p in self.parameters() if p not in fine_tune]
        
        # return [
        #     {'params': fine_tune.keys(), 'lr':lr, 'modifier': fine_tuning},
        #     {'params': normal, 'lr':lr, 'modifier': 1.0}
        # ]



base_options = '|'.join(pretrained.models.keys())

parameters = struct(
    base_name = param ("resnet18", help = "name of pretrained resnet to use options: " + base_options),
    features  = param (64, help = "fixed size features in new conv layers"),
    first     = param (3, help = "first layer of anchor boxes, anchor size = anchor_scale * 2^n"),
    last      = param (7, help = "last layer of anchor boxes"),

    anchor_scale = param (4, help = "anchor scale relative to box stride"),
    shared    = param (False, help = "share weights between network heads at different levels"),
    square    = param (False, help = "restrict box outputs (and anchors) to square"),
  )

def extra_layer(inp, features):
    layer = nn.Sequential(
        *([Conv(inp, features, 1)] if inp != features else []),
        Residual(basic_block(features, features)),
        Residual(basic_block(features, features)),
        Conv(features, features, stride=2)
    )

    layer.apply(init_weights)
    return layer



def split_at(xs, n):
    return xs[:n], xs[n:]


def anchor_sizes(start, end, anchor_scale=4, square=False):

    aspects = [1] if square else [1/2, 1, 2]
    scales = [1, pow(2, 1/3), pow(2, 2/3)]

    return [box.anchor_sizes(anchor_scale * (2 ** i), aspects, scales) for i in range(start, end + 1)]


def extend_layers(layers, size, features=32):

    features_in = pretrained.layer_sizes(layers)[-1]
    num_extra = max(0, size - len(layers))

    layers += [extra_layer(features_in if i == 0 else features, features) for i in range(0, num_extra)]
    return layers[:size]




def create_fcn(args, dataset_args):
    assert dataset_args.input_channels == 3

    num_classes = len(dataset_args.classes)
    assert num_classes >= 1

    assert args.first <= args.last
    assert args.base_name in pretrained.models, "base model not found: " + args.base_name + ", options: " + base_options

    base_layers = pretrained.models[args.base_name]()

    layer_names = ["layer" + str(n) for n in range(0, args.last + 1)]
    layers = extend_layers(base_layers, args.last + 1, features=args.features*2)

    backbone = Cascade(OrderedDict(zip(layer_names, layers)), drop_initial = args.first)
    box_sizes = anchor_sizes(args.first, args.last, anchor_scale=args.anchor_scale, square=args.square)


    return FCN(backbone, box_sizes, layer_names[args.first:], fine_tune=base_layers, num_classes=num_classes, features=args.features, shared=args.shared, square=args.square), \
           Encoder(args.first, box_sizes)

models = {
    'fcn' : struct(create=create_fcn, parameters=parameters)
  }



if __name__ == '__main__':

    _, *cmd_args = sys.argv

    model_params = {k: v.parameters for k, v in models.items()}
    parameters = struct(model= choice('fcn', model_params))

    parser = make_parser('Object detection', parameters)
    args = struct(**parser.parse_args().__dict__)

    model_args = parse_choice("model", parameters.model, args.model)

    classes = {
        0 : {'shape':'BoxShape'},
        1 : {'shape':'BoxShape'}
    }

    model, encoder = create_fcn(model_args.parameters, struct(classes = classes, input_channels = 3))

    x = Variable(torch.FloatTensor(4, 3, 370, 500))
    out = model.cuda()(x.cuda())

    [print(y.size()) for y in out]
