import numpy as np
import cv2

import torch

from tools import struct, tensors_to, shape, map_tensors
from tools.parameters import param, parse_args

from tools.image import cv
from main import load_model

from evaluate import evaluate_image
from detection import box, display, detection_table

from dataset.annotate import tagged
from export_model import export_onnx

# import torch.onnx

from time import time
import json

parameters = struct (
    model = param('',  required = True,     help = "model checkpoint to use for detection"),

    input = param('',    required = True,   help = "input video sequence for detection"),
    output = param(None, type='str',        help = "output annotated video sequence"),

    scale = param(None, type='float', help = "scaling of input"),


    log = param(None, type='str', help="output json log of detections"),

    start = param(0, help = "start frame number"),
    end = param(None, type='int', help = "start end number"),

    show = param(False, help='show progress visually'),

    backend = param('pytorch', help='use specific backend (onnx | pytorch | tensorrt)'),

    threshold = param(0.3, "detection threshold"),
    batch = param(8, "batch size for faster evaluation")
)

args = parse_args(parameters, "video detection", "video evaluation parameters")
print(args)

model, encoder, model_args = load_model(args.model)
print("model parameters:")
print(model_args)

classes = model_args.dataset.classes

frames, info  = cv.video_capture(args.input)
print(info)

output_size = (int(info.size[0] // 2), int(info.size[1] // 2))

scale = args.scale or 1
size = (int(info.size[0] * scale), int(info.size[1] * scale))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
nms_params = detection_table.nms_defaults._extend(threshold = args.threshold)

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


def evaluate_onnx(model, encoder, size, onnx_file):   
    export_onnx(model, size, onnx_file)
    
    import onnxruntime as ort

    print("Device: " + ort.get_device())
    ort_session = ort.InferenceSession(onnx_file)

    def f(image, nms_params=detection_table.nms_defaults):
        assert image.dim() == 3, "evaluate_onnx: expected image of 3d [H,W,C], got: " + str(image.shape)
        image_size = (image.shape[1], image.shape[0])
        image = image.unsqueeze(0)

        prediction = ort_session.run(None, {'input': image.numpy()})
        prediction = map_tensors(prediction, torch.squeeze, 0)

        return encoder.decode(image_size, prediction, nms_params=nms_params)
    return f


def evaluate_pytorch(model, encoder, device = torch.cuda.current_device()):
    model.to(device)
    encoder.to(device)

    def f(image, nms_params=detection_table.nms_defaults):
        return evaluate_image(model, frame, encoder, nms_params=nms_params, device=device).detections
    return f

def evaluate_tensorrt(model, encoder, device = torch.cuda.current_device()):
    print ("Compiling with tensorrt...")

    model.to(device)
    encoder.to(device)

    from torch2trt import torch2trt
    x = torch.ones(1, 3, int(size[1]), int(size[0])).to(device)
    model = torch2trt(model, [x], max_workspace_size=1<<26).to(device)

    def f(image, nms_params=detection_table.nms_defaults):
        return evaluate_image(model, frame, encoder, nms_params=nms_params, device=device).detections

    return f


device = torch.cuda.current_device()

evaluate = None
if args.backend == "tensorrt":
    evaluate = evaluate_tensorrt(model, encoder, device=device)
elif args.backend == "onnx":
    filename = args.model + ".onnx"
    evaluate = evaluate_onnx(model, encoder, size, filename)
elif args.backend == "pytorch":
    evaluate = evaluate_pytorch(model, encoder, device=device)
else:
    assert False, "unknown backend: " + args.backend

print("Ready")




detection_frames = []

start = time()
last = start

for i, frame in enumerate(frames()):
    if i > args.start:
        if scale != 1:
            frame = cv.resize(frame, size)

        detections = evaluate(frame, nms_params=nms_params)

        if args.log:
            detection_frames.append(export_detections(detections))

        if args.show or args.output:
            for prediction in detections._sequence():
                label_class = classes[prediction.label]
                display.draw_box(frame, prediction.bbox, confidence=prediction.confidence, 
                    name=label_class.name, color=(int((1.0 - prediction.confidence) * 255), 
                    int(255 * prediction.confidence), 0))

        if args.show:
            frame = cv.rgb_to_bgr(cv.resize(frame, output_size))
            cv.imshow(frame)
        
        if args.output:
            frame = cv.rgb_to_bgr(cv.resize(frame, output_size))
            out.write(frame.numpy())

    if args.end is not None and i >= args.end:
        break

    if i % 50 == 49:
        now = time()
        elapsed = now - last

        print("frame: {} 50 frames in {:.1f} seconds, at {:.2f} fps".format(i, elapsed, 50./elapsed))
        last = now

if out:
    out.release()

if args.log:
    with open(args.log, "w") as f:
        text = json.dumps(info._extend(filename=args.input, frames=detection_frames)._to_dicts())
        f.write(text)
