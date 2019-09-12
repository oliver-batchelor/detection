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
from tools import struct, table, show_shapes, sum_list, cat_tables, Struct

from tools.parameters import param, choice, parse_args, parse_choice, make_parser
from collections import OrderedDict

from detection.loss import batch_focal_loss


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


def default_decoder(features):
        decoder = nn.Sequential (
            Residual(basic_block(features, features)),
            Residual(basic_block(features, features))
        )
        # decoder = identity
        return Decode(features, module=decoder)

def default_subnet(features, n):
    return nn.Sequential(
        Residual(basic_block(features, features)),
        Residual(basic_block(features, features)),
        Conv(features, n, 1, bias=True))        


def join_outputs(layers, n):
    def permute(layer):
        out = layer.permute(0, 2, 3, 1).contiguous()
        return out.view(out.size(0), -1, n)

    return torch.cat(list(map(permute, layers)), 1)    

def outputs(layers, n):
    def permute(layer):
        out = layer.permute(0, 2, 3, 1).contiguous()
        return out.view(out.size(0), -1, n)

    return torch.cat(list(map(permute, layers)), 1)    

class PyramidOutput(nn.Module):
    def __init__(self, n_outputs, features, init=init_weights, joined=True):   
        super().__init__()
        modules = [(k, default_subnet(inputs, n_outputs)) for k, inputs in features.items()]
        
        for k, m in modules:
            m.apply(init_weights)

        self.subnets = Parallel(OrderedDict(modules))
        self.n_outputs = n_outputs
        self.joined = joined

    def forward(self, input):
        if self.joined:
            join_outputs(self.subnets(input), self.n_outputs)
        else:
            return self.subnets(input)


def regression(n, joined=True):
    def f(features):
        return JoinedOutput(n, features)
    return f


def classifier(n, joined=True): 
    def f(features):
        return JoinedOutput(n, features, init=init_classifier)
    return f


class FeaturePyramid(nn.Module):
    """ 
        backbone: The backbone network split into a list of layers acting at one resolution level (downsampling + processing layers)
        layers:   The range of output layers required.
        features: Number of features in the outputs and decoder side of the network

        outputs: A struct of functions to construct outputs given a list of feature map sizes. 
            e.g. struct(classes=classifier(2), boxes=regression(4, joined=True))
    """

    def __init__(self, outputs, backbone_layers, layer_range=(3, 7), features=32, make_decoder=default_decoder):
        super().__init__()
       
        def layer_names(r):
            return ["layer" + str(i) for i in range(r[0], r[1] + 1)]

        def named(modules, r = layer_range):
            names = layer_names(r)

            assert len(modules) == len(names)
            return OrderedDict(zip(names, modules))

        self.backbone = Cascade(named(backbone_layers, (0, layer_range[1])), drop_initial = layer_range[0])

        def make_reducer(size):
            return Conv(size, features, 1)
     
        encoded_sizes = pretrained.encoder_sizes(self.backbone)
        self.reduce = Parallel(named([make_reducer(size) for size in encoded_sizes]))
        self.decoder = UpCascade(named([make_decoder(features) for i in encoded_sizes]))

        output_features = OrderedDict([(i, features) for i in layer_names(layer_range)])
        self.outputs = nn.ModuleDict( {k :f(output_features) for k, f in outputs.items()} )

        for m in [self.reduce, self.decoder]:
            m.apply(init_weights)

    def forward(self, input):
        layers = self.backbone(input)
        layers = self.decoder(self.reduce(layers))

        output_dict = {k : output(layers) for k, output in self.outputs.items()}
        return Struct(output_dict)

base_options = '|'.join(pretrained.models.keys())

parameters = struct(
    backbone  = param ("resnet18", help = "name of pretrained model to use as backbone: " + base_options),
    features  = param (64, help = "fixed size features in new conv layers"),
    first     = param (3, help = "first layer of anchor boxes, anchor size = anchor_scale * 2^n"),
    last      = param (7, help = "last layer of anchor boxes"),
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



def extend_layers(layers, size, features=32):
    layer_sizes = pretrained.layer_sizes(layers)

    features_in = layer_sizes[-1]
    num_extra = max(0, size - len(layers))

    layers += [extra_layer(features_in if i == 0 else features, features) for i in range(0, num_extra)]
    return layers[:size]


def feature_pyramid(outputs, backbone_name, layer_range=(3, 7), features=32):

    assert layer_range[0] <= layer_range[1]
    assert backbone_name in pretrained.models, "base model not found: " + backbone_name + ", options: " + base_options

    base_layers = pretrained.models[backbone_name]()
    backbone_layers = extend_layers(base_layers, layer_range[1] + 1, features = features*2)

    return FeaturePyramid(outputs, backbone_layers, layer_range=layer_range, features=features)


if __name__ == '__main__':

    _, *cmd_args = sys.argv

    outputs = struct(classes=classifier(2), boxes=regression(4))
    model = feature_pyramid(outputs, 'resnet18')

    x = torch.FloatTensor(4, 3, 370, 500)
    out = model.cuda()(x.cuda())

    [print(k, y.size()) for k, y in out.items()]

