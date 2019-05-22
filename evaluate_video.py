import numpy as np
import cv2

import torch

from tools import struct
from tools.parameters import param, parse_args

from tools.image import cv
from main import load_model

from evaluate import evaluate_image
from detection import box, display

parameters = struct (
    model = param('',  required = True,     help = "model checkpoint to use for detection"),

    input = param('',    required = True,   help = "input video sequence for detection"),
    output = param(None, type='str',        help = "output annotated video sequence"),

    start = param(0, help = "start frame number"),
    end = param(None, type='int', help = "start end number"),

    threshold = param(0.5, "detection threshold"),
    batch = param(8, "batch size for faster evaluation")
)


args = parse_args(parameters, "video detection", "video evaluation parameters")
device = torch.cuda.current_device()

model, encoder, model_args = load_model(args.model)
print("model parameters:")
print(model_args)

classes = model_args.dataset.classes

model.to(device)

frames, info  = cv.video_capture(args.input)


for i, frame in enumerate(frames()):
    if i > args.start:

        nms_params = box.nms_defaults._extend(threshold = args.threshold)
        detections = evaluate_image(model, frame, encoder, nms_params = nms_params, device=device)
        
        for prediction in detections._sequence():
            label_class = classes[prediction.label].name
            display.draw_box(frame, prediction.bbox, confidence=prediction.confidence, name=label_class.name, color=display.to_rgb(label_class.colour))

       
        cv.imshow(frame)


    if args.end is not None and i >= args.end:
        break
