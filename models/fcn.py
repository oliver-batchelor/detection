import sys
import math

import torch
from torch import Tensor
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
from tools import struct, table, show_shapes, sum_list, cat_tables

from tools.parameters import param, choice, parse_args, parse_choice, make_parser
from collections import OrderedDict

from detection.loss import batch_focal_loss


def image_size(inputs):

    if torch.is_tensor(inputs):
        assert(inputs.dim() == 3)
        inputs = inputs.size(1), inputs.size(0)

    assert (len(inputs) == 2)
    return inputs


class Encoder:
    def __init__(self, start_layer, box_sizes, class_weights):
        self.anchor_cache = {}

        self.box_sizes = box_sizes
        self.start_layer = start_layer

        self.class_weights=class_weights


    def anchors(self, input_size, crop_boxes=False):
        def layer_size(i):
            stride = 2 ** i
            return (stride, max(1, math.ceil(input_size[0] / stride)), max(1, math.ceil(input_size[1] / stride)))

        input_args = (input_size, crop_boxes)

        if not (input_args in self.anchor_cache):
            layer_dims = [layer_size(self.start_layer + i) for i in range(0, len(self.box_sizes))]

            self.anchor_cache[input_args] = box.make_anchors(self.box_sizes, layer_dims, input_size, crop_boxes=crop_boxes)

        return self.anchor_cache[input_args]



    def encode(self, inputs, target, match_params=box.default_match):
        inputs = image_size(inputs)
        return box.encode(target, self.anchors(inputs, match_params.crop_boxes), match_params=match_params)


    def decode(self, inputs, prediction, crop_boxes=False):
        assert prediction.location.dim() == 2 and prediction.classification.dim() == 2

        inputs = image_size(inputs)
        anchor_boxes = self.anchors(inputs).type_as(prediction.location)

        bbox = box.decode(prediction.location, anchor_boxes)
        confidence, label = prediction.classification.max(1)

        if crop_boxes:
            box.clamp(bbox, (0, 0), inputs)

        return table(bbox = bbox, confidence = confidence, label = label)

       
    def loss(self, encoding, prediction, device):
 
       target = encoding._map(Tensor.to, device)
       return batch_focal_loss(target, prediction, averaging=False, class_weights=self.class_weights)
 

    def nms(self, prediction, nms_params=box.nms_defaults):
        return box.nms(prediction, nms_params)
    

def split_prediction_batch(prediction, i):
    return struct(
        classification = prediction.classification[:, :, i].unsqueeze(2),
        location = prediction.location[:, :, i, :].contiguous()
    )        

def split_prediction(prediction, i):
    return struct(
        classification = prediction.classification[:, i].unsqueeze(1),
        location = prediction.location[:, i, :].contiguous()
    )        



class SeparateEncoder:
    def __init__(self, start_layer, box_sizes, num_classes):
        self.encoder = Encoder(start_layer, box_sizes)
        self.num_classes = num_classes

    def anchors(self, input_size, crop_boxes=False):
        return self.encoder.anchors(input_size, crop_boxes=crop_boxes)

  
    def encode(self, inputs, target, crop_boxes=False, match_thresholds=(0.4, 0.5), match_nearest = 0):
        inputs = image_size(inputs)

        def encode_class(i):
            inds = (target.label == i).nonzero().squeeze(1)
            target_class = target._index_select(inds)

            target_class.label.fill_(0)

            return self.encoder.encode(inputs, target_class, 
                crop_boxes=crop_boxes, match_thresholds=match_thresholds, match_nearest=match_nearest)

        return [encode_class(i) for i in range(0, self.num_classes)]
        
    def loss(self, encoding, prediction, device):
        def class_loss(i):          
            return self.encoder.loss(encoding[i], split_prediction_batch(prediction, i), device)

        losses = [class_loss(i) for i in range(0, self.num_classes)]
        return sum_list(losses)

    def decode(self, inputs, prediction, crop_boxes=False):
        def decode_class(i):
            decoded = self.encoder.decode(inputs, split_prediction(prediction, i), crop_boxes=crop_boxes) 
            decoded.label.fill_(i)

            return decoded

        return cat_tables([decode_class(i) for i in range(0, self.num_classes)])


        
    def nms(self, prediction, nms_params=box.nms_defaults):
        return self.encoder.nms(prediction, nms_params=nms_params)




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

    def __init__(self, backbone, box_sizes, layer_names, fine_tune = [], features=32, num_classes=2, shared=False, square=False, separate=False):
        super().__init__()
       
        self.box_sizes = box_sizes
        self.num_classes = num_classes
        self.square = square
        self.separate = separate

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


        self.box_outputs = 4 
        self.box_layers = self.num_classes if self.separate else 1
        self.localisers = Parallel(named([output(len(boxes) * self.box_layers * self.box_outputs) for boxes in self.box_sizes]))


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

        locs = join(self.localisers(layers), self.box_outputs * self.box_layers)

        if self.separate:
            locs = locs.view(locs.size(0), locs.size(1), self.box_layers, self.box_outputs)
                
        # if self.square:
            # locs = locs.view(locs.size(0), locs.size(1), locs.size(2) )
            # locs = torch.cat([locs, locs.narrow(2, 2, 1)], dim=2)

        return struct( location = locs, classification = conf )

    def parameter_groups(self, lr, fine_tuning=0.1):
        
        return [
            {'params': self.backbone.parameters(), 'lr':lr, 'modifier': fine_tuning},
            {'params': nn.ModuleList(self.new_modules).parameters(), 'lr':lr, 'modifier': 1.0}
        ]




base_options = '|'.join(pretrained.models.keys())

parameters = struct(
    backbone  = param ("resnet18", help = "name of pretrained model to use as backbone: " + base_options),
    features  = param (64, help = "fixed size features in new conv layers"),
    first     = param (3, help = "first layer of anchor boxes, anchor size = anchor_scale * 2^n"),
    last      = param (7, help = "last layer of anchor boxes"),

    anchor_scale = param (4, help = "anchor scale relative to box stride"),

    tall = param(False, "use a set of anchor boxes for tall objects"),

    shared    = param (False, help = "share weights between network heads at different levels"),
    square    = param (False, help = "restrict box outputs (and anchors) to square"),

    separate =  param (False, help = "separate box location prediction for each class")
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


def anchor_sizes(start, end, anchor_scale=4, square=False, tall=False):

    aspects = [1/2, 1, 2]
    if square:
        aspects = [1]
    elif tall:
        aspects = [1/8, 1/4, 1/2, 1]

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
    assert args.backbone in pretrained.models, "base model not found: " + args.backbone + ", options: " + base_options

    assert not (args.square and args.tall), "model can be tall or square but not both"

    base_layers = pretrained.models[args.backbone]()

    layer_names = ["layer" + str(n) for n in range(0, args.last + 1)]
    layers = extend_layers(base_layers, args.last + 1, features=args.features*2)

    backbone = Cascade(OrderedDict(zip(layer_names, layers)), drop_initial = args.first)
    box_sizes = anchor_sizes(args.first, args.last, anchor_scale=args.anchor_scale, square=args.square, tall=args.tall)

    model = FCN(backbone, box_sizes, layer_names[args.first:], fine_tune=base_layers, 
                num_classes=num_classes, features=args.features, shared=args.shared, square=args.square, separate=args.separate)

    class_weights = [c.name.weighting for c in dataset_args.classes]

    encoder = SeparateEncoder(args.first, box_sizes, num_classes=num_classes)  \
        if args.separate else Encoder(args.first, box_sizes, class_weights=class_weights)

    return model, encoder
    

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
        0 : {'shape':'box'},
        1 : {'shape':'box'}
    }

    model, encoder = create_fcn(model_args.parameters, struct(classes = classes, input_channels = 3))

    x = Variable(torch.FloatTensor(4, 3, 370, 500))
    out = model.cuda()(x.cuda())

    [print(y.size()) for y in out]
