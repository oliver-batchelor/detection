import sys
import math

import torch
from torch import Tensor
import torch.nn as nn

import itertools
import torchvision.models as m

import models.pretrained as pretrained
from detection import box

from models.common import Conv, Cascade, UpCascade, Residual, Parallel, Shared, Lookup,  \
            DecodeAdd, Decode,  basic_block, se_block, reduce_features, replace_batchnorms, identity, GlobalSE

import torch.nn.init as init
from tools import struct, table, shape, sum_list, cat_tables, Struct

from tools.parameters import param, choice, parse_args, parse_choice, make_parser
from collections import OrderedDict



def init_weights(m):
    if isinstance(m, nn.Conv2d):
        init.normal_(m.weight, std=0.01)


def init_classifier(m, prior=0.001):
    if isinstance(m, nn.Conv2d):
        init.normal_(m.weight, std=0.01)
        if hasattr(m, 'bias') and m.bias is not None:           
            b = -math.log((1 - prior)/prior)
            init.constant_(m.bias, b)


def residual_decoder(num_blocks=2):
    def create(features):
        blocks = [Residual(basic_block(features, features))  for i in range(num_blocks)]
        decoder = nn.Sequential (*blocks)
        
        return Decode(features, module=decoder)
    return create


def residual_subnet(features, n, num_blocks=2):
    blocks = [Residual(basic_block(features, features))  for i in range(num_blocks)]
    return nn.Sequential (*blocks, Conv(features, n, 3), Conv(features, n, 1, bias=True))


def join_output(layers, n):
    def permute(layer):
        out = layer.permute(0, 2, 3, 1).contiguous()
        return out.view(out.size(0), -1, n)

    return torch.cat(list(map(permute, layers)), 1)    


class FeaturePyramid(nn.Module):
    """ 
        backbone_layers: The backbone network split into a list of layers acting at one resolution level (downsampling + processing layers)
        first:    Highest level resolution
        features: Number of features in the outputs and decoder side of the network
    """
    def __init__(self, backbone_layers, first=3, features=32, make_decoder=residual_decoder):
        super().__init__()

        backbone_names = list(backbone_layers.keys())
        self.backbone = Cascade(backbone_layers, drop_initial = first)

        self.names = backbone_names[first:]
        def named(modules):
            assert len(modules) == len(self.names)
            return OrderedDict(zip(self.names, modules))

        def make_reducer(size):
            return Conv(size, features, 1)

        self.first = first
        self.features = features
        self.depth = len(backbone_layers)
     
        encoded_sizes = pretrained.encoder_sizes(self.backbone)
        self.reduce = Parallel(named([make_reducer(size) for size in encoded_sizes]))
        self.decoder = UpCascade(named([make_decoder(features) for size in encoded_sizes]))

        for m in [self.reduce, self.decoder]:
            m.apply(init_weights)

    def forward(self, input):
        layers = self.backbone(input)
        return self.decoder(self.reduce(layers))

base_options = '|'.join(pretrained.models.keys())

pyramid_parameters = struct(
    backbone  = param ("resnet18", help = "name of pretrained model to use as backbone: " + base_options),
    features  = param (64, help = "fixed size features in new conv layers"),
    first     = param (3, help = "first layer of feature maps, scale = 1 / 2^first"),
    depth      = param (8, help = "depth in scale levels"),
    decode_blocks    = param(2, help = "number of residual blocks per layer in decoder")
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


def extend_layers(layers, size, features=32):
    layer_sizes = pretrained.layer_sizes(layers)

    features_in = layer_sizes[-1]
    num_extra = max(0, size - len(layers))

    layers += [extra_layer(features_in if i == 0 else features, features) for i in range(0, num_extra)]
    return layers[:size]


def label_layers(layers):
    layers = [(str(i), layer) for i, layer in enumerate(layers)]
    return OrderedDict(layers)

def feature_pyramid(backbone_name, first=3, depth=8, features=64, decode_blocks=2):

    assert first < depth
    assert backbone_name in pretrained.models, "base model not found: " + backbone_name + ", options: " + base_options

    base_layers = pretrained.models[backbone_name]()
    backbone_layers = label_layers(extend_layers(base_layers, depth, features = features*2))
    
    return FeaturePyramid(backbone_layers, first=first, features=features, make_decoder=residual_decoder(decode_blocks))

def feature_map(backbone_name, **options):
    pyramid = feature_pyramid(backbone_name, **options)
    return nn.Sequential(pyramid, Lookup(0))

if __name__ == '__main__':

    _, *cmd_args = sys.argv

    model = feature_pyramid('resnet18')

    x = torch.FloatTensor(4, 3, 500, 500)
    out = model.cuda()(x.cuda())

    print(shape(out))

