import torch

import os
import copy

from dataset.imports import load_dataset
from detection import models

from tools import struct, Struct

def show_differences(d1, d2, prefix=""):
    unequal_keys = []
    unequal_keys.extend(set(d1.keys()).symmetric_difference(set(d2.keys())))
    for k in d1.keys():
        if d1.get(k, '-') != d2.get(k, '-'):
            unequal_keys.append(k)
    if unequal_keys:
        for k in set(unequal_keys):
            v1 = d1.get(k, '-')
            v2 = d2.get(k, '-')
            if type(v1) != type(v2):
                v1 = type(v1)
                v2 = type(v2)

            print ("{:20s} {:10s}, {:10s}".format(prefix + k, str(v1), str(v2)))


def copy_partial(dest, src):
    assert src.dim() == dest.dim()

    for d in range(0, src.dim()):

        if src.size(d) > dest.size(d):
            src = src.narrow(d, 0, dest.size(d))
        else:
            dest = dest.narrow(d, 0, src.size(d))

    dest.copy_(src)

def load_state_partial(model, src):
    dest = model.state_dict()

    for k, dest_param in dest.items():
        if k in src:
            source_param = src[k]

            if source_param.dim() == dest_param.dim():
                copy_partial(dest_param, source_param)

def load_state(model, info, strict=True):    
    if strict:
        model.load_state_dict(info.state, strict=True)
    else:
        load_state_partial(model, info.state)

    return struct(model = model, 
        thresholds=info.thresholds if 'thresholds' in info else None, 
        score = info.score, epoch = info.epoch)

def new_state(model):
    return struct (model = model, score = 0.0, epoch = 0, thresholds = None)

def try_load(model_path):
    try:
        return torch.load(model_path)
    except (FileNotFoundError, EOFError, RuntimeError):
        pass

def load_model(model_path):
    loaded = try_load(model_path)
    assert loaded is not None, "failed to load model from " + model_path

    args = loaded.args

    model, encoder = models.create(args.model, args.dataset)
    load_state(model, loaded.best)

    return model, encoder, args


def load_checkpoint(model_path, model, model_args, args, strict=True):
    loaded = try_load(model_path)

    if not (args.no_load or not (type(loaded) is Struct)):

        current = load_state(model, loaded.best if args.restore_best else loaded.current, strict=strict)
        best = load_state(copy.deepcopy(model), loaded.best, strict=strict)

        print(loaded.args)

        if loaded.args == model_args:
            print("loaded model dataset parameters match, resuming")

        else:
            print("loaded model dataset parameters differ, loading partial")
            show_differences(model_args.__dict__,  loaded.args.__dict__)

            best.score = 0.0
            best.epoch = current.epoch
            best.thresholds = None

        return best, current, True

    return new_state(copy.deepcopy(model)), new_state(model), False
