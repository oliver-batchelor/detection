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
            DecodeAdd, Decode,  basic_block, reduce_features, replace_batchnorms

import torch.nn.init as init
from tools import Struct

from tools.parameters import param, choice, parse_args, parse_choice, make_parser


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


    def encode(self, inputs, boxes, labels, crop_boxes=False, match_thresholds=(0.4, 0.5)):
        inputs = image_size(inputs)
        return box.encode(boxes, labels, self.anchors(inputs, crop_boxes), match_thresholds)


    def decode(self, inputs, loc_pred, class_pred):
        assert loc_pred.dim() == 2 and class_pred.dim() == 2

        inputs = image_size(inputs)
        anchor_boxes = self.anchors(inputs).type_as(loc_pred)

        return box.decode(loc_pred, class_pred, anchor_boxes)

    def nms(self, boxes, labels, confs, nms_params=box.nms_defaults):
        return box.filter_nms(boxes, labels, confs, **nms_params)

    #
    # def decode_batch(self, inputs, loc_pred, class_pred, nms_params=box.nms_defaults):
    #     assert loc_pred.dim() == 3 and class_pred.dim() == 3
    #
    #     if torch.is_tensor(inputs):
    #         assert(inputs.dim() == 4)
    #         inputs = inputs.size(2), inputs.size(1)
    #
    #     assert len(inputs) == 2
    #     return [self.decode(inputs, l, c, nms_params=nms_params) for l, c in zip(loc_pred, class_pred)]





class FCN(nn.Module):

    def __init__(self, trained, extra, box_sizes, features=32, num_classes=2, shared=False, square=False):
        super().__init__()

        self.encoder = pretrained.make_cascade(trained + extra)
        self.box_sizes = box_sizes

        encoded_sizes = pretrained.encoder_sizes(self.encoder)
        self.reduce = Parallel([Conv(size, features, 1) for size in encoded_sizes])
        self.square = square

        def make_decoder():
            decoder = Residual(basic_block(features, features))
            return Decode(features, decoder)

        self.decoder = UpCascade([make_decoder() for i in encoded_sizes])

        assert len(trained + extra) == len(box_sizes), "FCN: layers and box sizes differ in length"

        def output(n):
            return nn.Sequential(
                Residual(basic_block(features, features)),
                Residual(basic_block(features, features)),
                Conv(features, n, 1, bias=True))

        self.num_classes = num_classes

        if shared:
            num_boxes = len(box_sizes[0])
            assert all(len(boxes) == num_boxes for boxes in box_sizes), "FCN: for shared heads, number of boxes must be equal on all layers"
            self.classifiers = Shared(output(num_boxes * self.num_classes))
        else:
            self.classifiers = Parallel([output(len(boxes) * self.num_classes) for boxes in self.box_sizes])

        self.localisers = Parallel([output(len(boxes) * (3 if square else 4)) for boxes in self.box_sizes])


        self.trained_modules = nn.ModuleList(trained)
        self.new_modules = nn.ModuleList(extra + [self.reduce, self.decoder, self.classifiers, self.localisers])

        def set_momentum(m):
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.2


        self.apply(set_momentum)


        def set_weights(m):
            # b = -math.log((1 - prior_prob)/prior_prob)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0)

        self.new_modules.apply(set_weights)

        #replace_batchnorms(self, 16)



    def forward(self, input):
        layers = self.decoder(self.reduce(self.encoder(input)))
        def join(layers, n):

            def permute(layer):
              out = layer.permute(0, 2, 3, 1).contiguous()
              return out.view(out.size(0), -1, n)

            return torch.cat(list(map(permute, layers)), 1)

        # def make_rect(layer):
        #     # Duplicate the size output for x and y together
        #     layers = [layer, layer.narrow(1, 2, 1)]
        #     print(layer.size(), layer.narrow(1, 2, 1).size())
        #
        #     return torch.cat(layers, dim=1)

        conf = torch.sigmoid(join(self.classifiers(layers), self.num_classes))
        locs = join(self.localisers(layers), 3 if self.square else 4)

        if self.square:
            locs = torch.cat([locs, locs.narrow(2, 2, 1)], dim=2)


        return (locs, conf)

    def parameter_groups(self, lr, fine_tuning=0.1):
        return [
            {'params': self.trained_modules.parameters(), 'lr':lr, 'modifier': fine_tuning},
            {'params': self.new_modules.parameters(), 'lr':lr, 'modifier': 1.0}
        ]


base_options = '|'.join(pretrained.models.keys())

parameters = Struct(
        base_name = param ("resnet18", help = "name of pretrained resnet to use options: " + base_options),
        features  = param (64, help = "fixed size features in new conv layers"),
        first     = param (3, help = "first layer of anchor boxes, anchor size = anchor_scale * 2^n"),
        last      = param (5, help = "last layer of anchor boxes"),

        anchor_scale = param (4, help = "anchor scale relative to box stride"),
        shared    = param (False, help = "share weights between network heads at different levels"),
        square    = param (False, help = "restrict box outputs (and anchors) to square"),

    )

def extra_layer(inp, features):
    return nn.Sequential(
        *([Conv(inp, features, 1)] if inp != features else []),
        Residual(basic_block(features, features)),
        Residual(basic_block(features, features)),
        Conv(features, features, stride=2)
    )

def split_at(xs, n):
    return xs[:n], xs[n:]


def anchor_sizes(start, end, anchor_scale=4, square=False):

    aspects = [1] if square else [1/2, 1, 2]
    scales = [1, pow(2, 1/3), pow(2, 2/3)]

    return [box.anchor_sizes(anchor_scale * (2 ** i), aspects, scales) for i in range(start, end + 1)]


def extend_layers(layers, start, end, features=32):
    features_in = pretrained.layer_sizes(layers)[-1]

    num_extra = max(0, end + 1 - len(layers))
    extra_layers = [extra_layer(features_in if i == 0 else features, features) for i in range(0, num_extra)]

    initial, rest =  layers[:start + 1], layers[start + 1:end + 1:]
    return [nn.Sequential(*initial), *rest], [*extra_layers]


def create_fcn(args, dataset_args):
    assert dataset_args.input_channels == 3

    num_classes = len(dataset_args.classes)
    assert num_classes >= 1

    assert args.first <= args.last
    assert args.base_name in pretrained.models, "base model not found: " + args.base_name + ", options: " + base_options

    base_layers = pretrained.models[args.base_name]()
    args.first = min(len(base_layers) - 1, args.first)

    backbone, extra = extend_layers(base_layers, args.first, args.last, features=args.features)
    box_sizes = anchor_sizes(args.first, args.last, anchor_scale=args.anchor_scale, square=args.square)

    return FCN(backbone, extra, box_sizes, num_classes=num_classes, features=args.features, shared=args.shared, square=args.square), \
           Encoder(args.first, box_sizes)

models = {
    'fcn' : Struct(create=create_fcn, parameters=parameters)
  }




if __name__ == '__main__':

    _, *cmd_args = sys.argv

    model_params = {k: v.parameters for k, v in models.items()}
    parameters = Struct(model= choice('fcn', model_params))

    parser = make_parser('Object detection', parameters)
    args = Struct(**parser.parse_args().__dict__)

    model_args = parse_choice("model", parameters.model, args.model)

    classes = {
        0 : {'shape':'BoxShape'},
        1 : {'shape':'BoxShape'}
    }

    model, _ = create_fcn(model_args.parameters, Struct(classes = classes, input_channels = 3))

    x = Variable(torch.FloatTensor(4, 3, 400, 400))
    out = model.cuda()(x.cuda())

    [print(y.size()) for y in out]
