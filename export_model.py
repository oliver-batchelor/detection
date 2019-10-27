import numpy as np
import cv2

from time import time

from tools import struct, shape
from tools.parameters import param, parse_args

from tools.image import cv, transforms
from main import load_model

import torch
import torch.nn as nn
import torch.onnx
import onnx

from onnx import optimizer
from onnx import helper, shape_inference


from time import time
import json

parameters = struct (
    model = param('',  required = True,     help = "model checkpoint to use for detection"),
    size = param('1920x1080', help = "input resolution"),

    onnx_file = param(type='str',  required=True,   help = "output file"),
)

def export_onnx(model, size, filename):
    model_aug = nn.Sequential(transforms.Normalize(), model).cpu()

    dummy = torch.ByteTensor(1, int(size[1]), int(size[0]), 3)
    torch.onnx.export(model_aug,               # model being run
                    dummy,                         # model input (or a tuple for multiple inputs)
                    filename,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    # opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # wether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['location', 'classification'], # the model's output names
                    dynamic_axes={})

    onnx_model = onnx.load(filename)
    onnx.checker.check_model(onnx_model)


    # graph = onnx.helper.printable_graph(model.graph)
    # print(graph)


    # inferred_model = shape_inference.infer_shapes(model)
    # onnx.checker.check_model(inferred_model)

    # print(model.graph.value_info, inferred_model.graph.value_info)

    # all_passes = optimizer.get_available_passes()
    # optimized = optimizer.optimize(model, all_passes)

    # graph = onnx.helper.printable_graph(model.graph)
    # print(graph)

if __name__=='__main__':
    args = parse_args(parameters, "export model", "export parameters")
    print(args)
    device = torch.cuda.current_device()
    # device = torch.device('cpu')

    model, encoder, model_args = load_model(args.model)
    print("model parameters:")
    print(model_args)

    size = args.size.split("x")
