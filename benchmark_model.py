import numpy as np
import cv2

import torch

from tools import struct
from tools.parameters import param, parse_args

from tools.image import cv
from main import load_model

from evaluate import evaluate_image
from detection import box, display, detection_table

from dataset.annotate import tagged

from time import time
import json

parameters = struct (
    model = param('',  required = True,     help = "model checkpoint to use for detection"),
    input = param('',    required = True,   help = "input video sequence for detection"),

    scale = param(None, type='float', help = "scaling of input"),

    tensorrt = param(False, help='optimize model with tensorrt'),

    frames = param(256, help="number of frames to use"),

    threshold = param(0.3, "detection threshold"),
    batch = param(8, "batch size for faster evaluation")
)

args = parse_args(parameters, "model benchmark", "parameters")
print(args)
device = torch.cuda.current_device()

model, encoder, model_args = load_model(args.model)
print("model parameters:")
print(model_args)

classes = model_args.dataset.classes

model.to(device)
encoder.to(device)

frames, info  = cv.video_capture(args.input)
print(info)

scale = args.scale or 1
size = (int(info.size[0] * scale), int(info.size[1] * scale))

nms_params = detection_table.nms_defaults._extend(threshold = args.threshold)
images = []

def print_timer(desc, frames, start):
    elapsed = time() - start
    print("{}: {} frames in {:.1f} seconds, at {:.2f} fps".format(desc, frames, elapsed, frames / elapsed))

start = time()

for i, frame in enumerate(frames()):
    if scale != 1:
        frame = cv.resize(frame, size)

    images.append(frame)
    if len(images) >= args.frames:
        break

fps = len(images) / (time() - start)
print_timer("load", len(images), start)

if args.tensorrt:
    print ("compiling with tensorrt...")
    from torch2trt import torch2trt
    x = torch.ones(1, 3, int(size[1]), int(size[0])).to(device)
    model = torch2trt(model, [x], fp16_mode=True)
    print("done")

dummy = torch.ones(1, 3, int(size[1]), int(size[0])).to(device)
model(dummy)

start = time()

for i in range(len(images)):  
    dummy = torch.ones(1, 3, int(size[1]), int(size[0])).to(device)
    model(dummy)

print_timer("model only", len(images), start)

start = time()

for image in images:        
    detections = evaluate_image(model, image, encoder, nms_params = nms_params, device=device).detections

print_timer("evaluate_image", len(images), start)

        