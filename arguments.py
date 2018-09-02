
import argparse

from tools.parameters import param, make_parser, choice, parse_choice, group
from tools import Struct

import detection.models as models

train_parameters = Struct (
    optimizer = group('optimizer settings',
        lr              = param(1.0,    help='learning rate'),
        lr_epoch_decay  = param(0.1,    help='decay lr during epoch by factor'),
        fine_tuning     = param(0.1,    help='fine tuning as proportion of learning rate'),
        momentum        = param(0.5,    help='SGD momentum'),
    ),

    seed            = param(1,      help='random seed'),
    batch_size      = param(4,     help='input batch size for training'),
    epoch_size      = param(1024,   help='epoch size for training'),

    num_workers     = param(4,      help='number of workers used to process dataset'),
    model           = choice(default='fcn', options=models.parameters, help='model type and parameters e.g. "fcn --start=4"'),

    no_load         = param(False,   help="don't attempt to load previously saved model"),
    run_name        = param('training', help='name for training run')
)


detection_parameters = Struct (
    image = group('image',
        min_scale   = param(2/3,   help='minimum scaling during preprocessing'),
        max_scale   = param(3/2,   help='maximum scaling during preprocessing'),
        gamma       = param(0.15,  help='variation in gamma (brightness) when training'),
        channel_gamma       = param(0.1,  help='variation per channel gamma when training'),
        image_size  = param(440,   help='size of patches to train on'),
        down_scale  = param(1,     help='down scale of image_size to test/train on'),

        full_size   = param(False, help='train always on full size images rather than sampling a patch'),
        transposes  = param(False, help='enable image transposes in training'),
        flips          = param(False, help='enable horizontal image flips in training'),
        vertical_flips = param(False, help='enable vertical image flips in training'),

        image_samples   = param(1,      help='number of training samples to extract from each loaded image')
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

    min_visible     = param (0.4, help = 'minimum proportion of area for an overlapped box to be included'),
    crop_boxes      = param(False, help='crop boxes to the edge of the image patch in training'),
)

parameters = detection_parameters.merge(train_parameters)

parser = make_parser('Object detection', parameters)

parser.add_argument('--input', default=None, help='input path to dataset')
parser.add_argument('--dataset', default='annotate', help='dataset type options are (annotate)')

parser.add_argument('--remote', default=None, help='host for connection to annotate server')

def get_arguments():
    args = Struct(**parser.parse_args().__dict__)

    if args.model:
         args.model = parse_choice("model", parameters.model, args.model)

    return args
