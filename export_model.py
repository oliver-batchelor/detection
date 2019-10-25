import numpy as np
import cv2

import torch

from tools import struct, shape
from tools.parameters import param, parse_args

from tools.image import cv
from main import load_model

import torch.onnx
import onnx

from onnx import optimizer
from onnx import helper, shape_inference

import onnxruntime as ort

from time import time
import json

parameters = struct (
    model = param('',  required = True,     help = "model checkpoint to use for detection"),
    size = param('1920x1080', help = "input resolution"),

    onnx_file = param(type='str',  required=True,   help = "output file"),
)

args = parse_args(parameters, "export model", "export parameters")
print(args)
device = torch.cuda.current_device()
# device = torch.device('cpu')

model, encoder, model_args = load_model(args.model)
print("model parameters:")
print(model_args)



model.to(device)
encoder.to(device)

size = args.size.split("x")

dummy = torch.ones(1, 3, int(size[1]), int(size[0])).to(device)
torch.onnx.export(model,               # model being run
                dummy,                         # model input (or a tuple for multiple inputs)
                args.onnx_file,   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=10,          # the ONNX version to export the model to
                do_constant_folding=True,  # wether to execute constant folding for optimization
                input_names = ['input'],   # the model's input names
                output_names = ['location', 'classification'], # the model's output names
                dynamic_axes={})

model = onnx.load(args.onnx_file)
onnx.checker.check_model(model)


ort_session = ort.InferenceSession(args.onnx_file)
outputs = ort_session.run(None, {'input': dummy.cpu().numpy()})

print(shape(outputs))


    # graph = onnx.helper.printable_graph(model.graph)
    # print(graph)


    # inferred_model = shape_inference.infer_shapes(model)
    # onnx.checker.check_model(inferred_model)

    # print(model.graph.value_info, inferred_model.graph.value_info)

    # all_passes = optimizer.get_available_passes()
    # optimized = optimizer.optimize(model, all_passes)

    # graph = onnx.helper.printable_graph(model.graph)
    # print(graph)