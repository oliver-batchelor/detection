
import torch
from torch import Tensor
import arguments

# import pyximport; pyximport.install()

from dataset.imports import load_dataset
from dataset.detection import get_match_params
from detection.display import overlay_batch

from arguments import detection_parameters, train_parameters, make_input_parameters, debug_parameters
from tools.parameters import parse_args

from tools import struct, Table, show_shapes, pluck, transpose_structs
from tools.parameters import param, required, parse_args, choice, parse_choice, make_parser
from tools.image import cv
from tools.image.transforms import resize_scale

import evaluate
from detection.evaluate import match_boxes, mAP_matches

import pprint
import random
import main

import detection.models as models
from tools.model import tools

import detection.box as box
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=2)


vis_parameters = struct (
    batch_size  = param(1,           help='batch size to display'),
    test        = param(False,       help='show test images instead of training'),
    validate        = param(False,       help='show validation images instead of training'),   
    no_augment  = param(False,    help='dont use preprocessing even for training'),

    best = param (False, help='use best model for evaluation'),
    action = param ("visualise",    help='action to take (visualise|evaluate|benchmark)')
)



def find_anchors(image, target, encoder, num_classes, match_params=box.default_match):
    size = (image.size(1), image.size(0))
    anchors = box.point_form(encoder.anchors(size, crop_boxes = match_params.crop_boxes))

    target_enc = encoder.encode(image, target, match_params=match_params)
    matches = []

    for i in range(0, num_classes):
        inds = target_enc.classification.eq(i + 1).nonzero().squeeze()

        if inds.dim() > 0:
            for b in anchors[inds]:
                matches.append(struct(label = i, bbox = b))

    return matches


def image_stats(batch):
    assert(batch.dim() == 3 and batch.size(2) == 3)

    batch = batch.float().div_(255)
    flat = batch.view(-1, 3)

    return struct(mean=flat.mean(0).cpu(), std=flat.std(0).cpu())


def identity(batch):
    return batch



def evaluate_vis(model, encoder, data, nms_params, args, iou = 0.5):

    with torch.no_grad():
        model.to(device)

        raw_prediction = evaluate.evaluate_raw(model, data.image, device=device)
        decoded = encoder.decode(data.image, raw_prediction, crop_boxes=args.crop_boxes)

        prediction = encoder.nms(decoded, nms_params=nms_params)

        # prediction = evaluate.evaluate_image(model, data.image, encoder, nms_params, device)

        target = data.target._map(Tensor.to, prediction._device)

        matches = match_boxes(prediction, target, threshold = iou)
        anchors = find_anchors(data.image, data.target, encoder, len(dataset.classes), match_params=get_match_params(args))

        result = mAP_matches(matches, target.label.size(0))


        # pred = transpose_structs(matches) 
        # bbox = torch.stack (pred.bbox)

        # detected = box.match_predictions(bbox, raw_prediction, threshold = 0.5)

        return struct(
            image = data.image,
            file = data.file,
            id = data.id,
            image_size = data.image_size,
            matches = matches,
            anchors = anchors,
            target = data.target,
            prediction = prediction,
            stats = image_stats(data.image),
            mAP = result.mAP
        )

def benchmark(model, encoder, iter, args):

    test = dataset.validate(args,  collate_fn=identity)
    train = dataset.sample_train(args, encoder=encoder)
   
    
    print ("load(training): ", len(train))
    for data in tqdm(train):
        pass
    print ("load(validate):", len(validate))
    for _ in tqdm(validate):
        pass


def visualise(model, encoder, iter, args):

    threshold = 50
    zoom = 100

    mode = 'target'

    help_str = """
    keys:
    -+:  reduce and increase threshold
    []:  zoom in and out

    (): adjust nms threshold
    <>: adjust max detections

    m:  show matches
    p:  show predictions
    a:  show matched anchor boxes
    t:  show target boxes
    """

    keys = struct(
        escape = 27,
        space = 32,
        minus = 45,
        plus = 61)

    nms = struct (
        nms = 0.5,
        threshold = 0.05,
        detections = 400
    )


    print(help_str)

    for batch in iter:

        key = 0
        while (not key in [keys.escape, keys.space]):

            evals = [evaluate_vis(model, encoder, i, nms, args) for i in batch]

            def show2(x):
                return "{:.2f}".format(x)

            def show_vector(v):
                return "({})".format(",".join(map(show2, v.tolist())))

            for e in evals:
                print("{}: {:d}x{:d} {}".format(e.file, e.image_size[0], e.image_size[1], str(e.stats._map(show_vector))))


            display = overlay_batch(evals, mode=mode, scale=100 / zoom, classes=dataset.classes, threshold = threshold/100, cols=4)
            if zoom != 100:
                display = resize_scale(display, zoom / 100)


            key = cv.display(display)

            if key == keys.minus and threshold > 0:
                threshold -= 5
                print("reducing threshold: ", threshold)
            elif key == keys.plus and threshold < 100:
                threshold += 5
                print("increasing threshold: ", threshold)

            elif chr(key) == '(' and nms.nms > 0:
                nms.nms = max(0, nms.nms - 0.05) 
                print("decreasing nms threshold: ", nms.nms)

            elif chr(key) == ')' and nms.nms < 1:
                nms.nms = min(1, 0.05 + nms.nms)
                print("increasing nms threshold: ", nms.nms)   
            elif chr(key) == '<' and nms.max_detections > 50:
                nms.detections -= 50
                print("decreasing max detections: ", nms.detections)
            elif chr(key) == '>':
                nms.detections += 50
                print("increasing max detections: ", nms.detections)

            elif key == 91 and zoom > 10:
                zoom -= 5
                print("zoom out: ", zoom)
            elif key == 93 and zoom < 200:
                zoom += 5
                print("zoom in: ", zoom)
            elif chr(key) == 'm':
                mode = 'matches'
                print("showing matches")
            elif chr(key) == 'p':
                mode = 'prediction'
                print("showing prediction")
            elif chr(key) == 'a':
                mode = 'anchors'
                print("showing anchors")
            elif chr(key) == 't':
                mode = 'target'
                print("showing target")


        if(key == 27):
            break


if __name__ == '__main__':
    device = torch.cuda.current_device()

    input_parameters = make_input_parameters()
    parameters = detection_parameters._merge(train_parameters)._merge(vis_parameters)._merge(input_parameters)._merge(debug_parameters)


    args = parse_args(parameters, "Visualise", "")

    args.model = parse_choice("model", parameters.model, args.model)
    args.input = parse_choice("input", parameters.input, args.input)

    args.dry_run = True
    args.no_load = False


    random.seed(args.seed)
    torch.manual_seed(args.seed)


    pp.pprint(args._to_dicts())

    config, dataset = load_dataset(args)
    env = main.initialise(config, dataset, args)

    iter = None

    if args.test:
        iter = dataset.test(args, encoder=None, collate=identity)
    elif args.validate:
        iter = dataset.validate(args, encoder=None, collate=identity)
    elif args.no_augment:
        iter = dataset.test_on(dataset.train_images, args, encoder=None, collate=identity)
    else:
        iter = dataset.sample_train(args, encoder=None, collate=identity)

    model = env.best.model if args.best else env.model

    def show_dim(x, y):
        return "{:d}x{:d}".format(int(x), int(y))

    print("anchor box sizes:")
    for dims in env.encoder.box_sizes:
        print(*("{0: <8}".format(show_dim(x, y)) for x, y in dims))


    if args.action == 'visualise':
        visualise(model, env.encoder, iter, args)
    elif args.action == 'evaluate':
        main.run_testing(model.to(device), env, device=device)
    elif args.action == 'benchmark':
        benchmark(model, env.encoder, iter, args)
    else:
        assert False, "unknown action: " + action

    print("done")
