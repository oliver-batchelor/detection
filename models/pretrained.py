import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import itertools
from torchvision.models import resnet, densenet, vgg

from pretrainedmodels.models import senet
import pretrainedmodels



import models.common as c

from tools import Struct


def make_encoder(name, depth = None):
    layers = get_layers(name)
    if depth:
        layers = layers[:depth]

    return make_cascade(layers)


def get_layers(name):
    settings = pretrainedmodels.pretrained_settings[name]
    model = pretrainedmodels.__dict__[name](pretrained='imagenet')

    if isinstance(model, senet.SENet):
        return senet_layers(model)
    if isinstance(model, resnet.ResNet):
        return resnet_layers(model)
    elif isinstance(model, densenet.DenseNet):
        return densenet_layers(model)
    elif isinstance(model, vgg.VGG):
        return vgg_layers(model)
    else:
        assert false, "unsupported model type " + name



def make_cascade(layers):
    return c.Cascade(*layers)

def layer_sizes(layers):
    return encoder_sizes(make_cascade(layers))

def encoder_sizes(encoder):
    encoder.eval()

    x = Variable(torch.FloatTensor(8, 3, 224, 224))
    skips = encoder(x)

    return [t.size(1) for t in skips]


def senet_layers(model):
    *layer0_modules, pool0 = model.layer0

    layer0 = nn.Sequential(*layer0_modules)
    layer1 = nn.Sequential(pool0, model.layer1)

    layers = [c.Identity(), layer0, layer1, model.layer2, model.layer3, model.layer4]
    return layers


def resnet_layers(model):
    layer0 = nn.Sequential(model.conv1, model.bn1, nn.ReLU(inplace=True))
    layer1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), model.layer1)

    layers = [c.Identity(), layer0, layer1, model.layer2, model.layer3, model.layer4]
    return layers


def vgg_layers(model):
    layers = []
    current = nn.Sequential()

    for i, m in enumerate(model.features._modules.values()):
        if isinstance(m, nn.MaxPool2d):
            layers.append(current)
            current = nn.Sequential(m)
        else:
            current.add_module(str(i), m)

    return layers

def densenet_layers(densenet):
    m = densenet.features._modules

    return [
            c.Identity(),
            nn.Sequential(m['conv0'], m['norm0'], m['relu0']),
            nn.Sequential(m['pool0'], m['denseblock1']),
            nn.Sequential(m['transition1'], m['denseblock2']),
            nn.Sequential(m['transition2'], m['denseblock3']),
            nn.Sequential(m['transition3'], m['denseblock4'], m['norm5'])
        ]
