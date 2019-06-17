import numpy as np
import cv2

import torch

from tools import struct
from tools.parameters import param, parse_args

from tools.image import cv
from main import load_model

from evaluate import evaluate_image
from detection import box, display

from dataset.annotate import tagged

import json

parameters = struct (
    model = param('',  required = True,     help = "model checkpoint to use for detection"),

    input = param('',    required = True,   help = "input video sequence for detection"),
    output = param(None, type='str',        help = "output annotated video sequence"),

    log = param(None, type='str', help="output json log of detections"),

    start = param(0, help = "start frame number"),
    end = param(None, type='int', help = "start end number"),

    show = param(False, help='show progress visually'),

    threshold = param(0.3, "detection threshold"),
    batch = param(8, "batch size for faster evaluation")
)

args = parse_args(parameters, "video detection", "video evaluation parameters")
print(args)
device = torch.cuda.current_device()

model, encoder, model_args = load_model(args.model)
print("model parameters:")
print(model_args)


classes = model_args.dataset.classes

model.to(device)
frames, info  = cv.video_capture(args.input)

print(info)

size = (int(info.size[0] // 2), int(info.size[1] // 2))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = None
if args.output:
    out = cv2.VideoWriter(args.output, fourcc, info.fps, size)



def encode_shape(box, config):
    lower, upper =  box[:2], box[2:]

    if config.shape == 'circle':

        centre = ((lower + upper) * 0.5).tolist()
        radius = ((upper - lower).sum().item() / 4)

        circle_shape = struct(centre = centre, radius = radius)
        return 'circle', circle_shape

    elif config.shape == 'box':
        return 'box', struct (lower = lower.tolist(), upper = upper.tolist())

    assert False, "unsupported shape config: " + config.shape



def export_detections(predictions):
    def detection(p):
        object_class = classes[p.label]
        config = object_class.name

        t, box = encode_shape(p.bbox.cpu(), config)
        return struct (
            box      = box, 
            label      =  p.label,
            confidence = p.confidence.item(),
            match = p.match.item() if 'match' in p else None
        )
        
    return list(map(detection, predictions._sequence()))

detection_frames = []

for i, frame in enumerate(frames()):
    if i > args.start:

        nms_params = box.nms_defaults._extend(threshold = args.threshold)
        detections = evaluate_image(model, frame, encoder, nms_params = nms_params, device=device)
        
        for prediction in detections._sequence():
            label_class = classes[prediction.label].name
            display.draw_box(frame, prediction.bbox, confidence=prediction.confidence, name=label_class.name, color=(int((1.0 - prediction.confidence) * 255), int(255 * prediction.confidence), 0))

        detection_frames.append(export_detections(detections))

        if args.show:
            cv.imshow(frame)
        frame = cv.rgb_to_bgr(cv.resize(frame, size))

        if out:
            out.write(frame.numpy())

    if args.end is not None and i >= args.end:
        break

    if i % 50 == 0:
        print(i)

if out:
    out.release()

if args.log:
    with open(args.log, "w") as f:
        text = json.dumps(info._extend(filename=args.input, frames=detection_frames)._to_dicts())
        f.write(text)
