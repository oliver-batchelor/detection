
import argparse

from tools.parameters import param, parse_args, choice, parse_choice, group
from tools import Struct

import detection.models as models

train_parameters = Struct (
    optimizer = group('optimizer settings',
        lr              = param(0.1,    help='learning rate'),
        lr_epoch_decay  = param(0.1,    help='decay lr during epoch by factor'),
        fine_tuning     = param(1.0,    help='fine tuning as proportion of learning rate'),
        momentum        = param(0.5,    help='SGD momentum'),
    ),

    average_start  = param(2,    help='start weight averaging after epochs'),
    average_window = param(1,    help='use a window of size for averaging, 1 = no averaging'),


    seed            = param(1,      help='random seed'),
    batch_size      = param(4,     help='input batch size for training'),
    epoch_size      = param(1024,   help='epoch size for training'),

    num_workers     = param(4,      help='number of workers used to process dataset'),
    model           = choice(default='fcn', options=models.parameters, help='model type and parameters e.g. "fcn --start=4"'),

    bn_momentum    = param(0.1, "momentum for batch normalisation modules"),

    no_load         = param(False,   help="don't attempt to load previously saved model"),
    dry_run         = param(False,   help="run for testing only (don't store results or log progress)"),

    restore_best    = param(False,   help="restore weights from best validation model"),

    run_name        = param('training', help='name for training run')
)


detection_parameters = Struct (
    image = group('image',
        min_scale   = param(2/3,   help='minimum scaling during preprocessing'),
        max_scale   = param(3/2,   help='maximum scaling during preprocessing'),
        gamma       = param(0.15,  help='variation in gamma (brightness) when training'),
        channel_gamma       = param(0.1,  help='variation per channel gamma when training'),
        image_size  = param(440,   help='size of patches to train on'),
        down_scale  = param(1.0,     help='down scale of image_size to test/train on'),

        full_size   = param(False, help='train always on full size images rather than sampling a patch'),
        transposes  = param(False, help='enable image transposes in training'),
        flips          = param(True, help='enable horizontal image flips in training'),
        vertical_flips = param(False, help='enable vertical image flips in training'),

        image_samples   = param(1,      help='number of training samples to extract from each loaded image'),

        allow_empty = param(0.2,    help='when image crop is empty, train with probability')
    ),

    match = group('match thresholds',
        pos_match = param (0.5, help = "lower iou threshold matching positive anchor boxes in training"),
        neg_match = param (0.4,  help = "upper iou threshold matching negative anchor boxes in training")
    ),

    nms = group('nms',
        nms_threshold    = param (0.5, help = "overlap threshold (iou) used in nms to filter duplicates"),
        class_threshold  = param (0.05, help = 'hard threshold used to filter negative boxes'),
        max_detections    = param (100,  help = 'maximum number of detections (for efficiency) in testing')
    ),

    min_visible     = param (0.2, help = 'minimum proportion of area for an overlapped box to be included'),
    crop_boxes      = param(False, help='crop boxes to the edge of the image patch in training'),
)


input_choices = Struct(
    json = Struct(
        path          = param(type="str", help = "path to exported json annotation file", required=True)),
    coco = Struct(
        path          = param("/home/oliver/storage/coco", help = "path to exported json annotation file")),
    voc = Struct(
        path          = param("/home/oliver/storage/voc", help = "path to exported json annotation file"),
        preset        = param("test2007", help='preset configuration of testing/training set used options test2007|val2012')
    )
)

def make_input_parameters(default = None, choices = input_choices):
    return Struct(
        keep_classes = param(type="str", help = "further filter the classes, but keep empty images"),
        subset       = param(type="str", help = "use a subset of loaded classes, filter images with no anntations"),
        input        = choice(default=default, options=choices, help='input method'),
    )




input_remote = make_input_parameters('remote', input_choices._extend(
    remote = Struct (host = param("localhost:2160", help = "hostname of remote connection"))
))

parameters = detection_parameters._merge(train_parameters)._merge(input_remote)


def get_arguments():
    args = parse_args(parameters, "trainer", "object detection parameters")

    args.model = parse_choice("model", parameters.model, args.model)
    args.input = parse_choice("input", parameters.input, args.input)


    return args
