from tools import struct
from .retina.model import model as retina
from .ttf.model import model as ttf

def merge(*dicts):
    m = {}
    for d in dicts:
        m.update(d)

    return m


models = struct(retina=retina, ttf=ttf)
parameters = models._map(lambda m: m.parameters)


def model_stats(model):
    convs = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            convs += 1

    parameters = sum([p.nelement() for p in model.parameters()])
    print("Model of {} parameters, {} convolutions".format(parameters, convs))


def create(model_args, dataset_args):
    assert model_args.choice in models, "model not found " + model_args.choice
    model = models[model_args.choice]
    return model.create(model_args.parameters, dataset_args)
