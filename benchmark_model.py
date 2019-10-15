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

    frames = param(128, help="number of frames to use"),

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

size = (int(info.size[0] // 2), int(info.size[1] // 2))

nms_params = detection_table.nms_defaults._extend(threshold = args.threshold)
images = []

for i, frame in enumerate(frames()):
    images.append(frame)
    if len(images) >= args.frames:
        break

print("loaded {} images".format(len(images)))

start = time()

for i in range(len(images)):  
    dummy = torch.tensor(1, 3, info.size[0], info.size[1])
    model(dummy)

now = time()
elapsed = now - start

print("model only: {} frames in {:.1f} seconds, at {:.2f} fps".format(len(images), elapsed, len(images)/elapsed))


start = time()

for image in images:        
    detections = evaluate_image(model, image, encoder, nms_params = nms_params, device=device).detections

now = time()
elapsed = now - start

print("evaluate_image: {} frames in {:.1f} seconds, at {:.2f} fps".format(len(images), elapsed, len(images)/elapsed))
        