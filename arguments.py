
import argparse

from tools.parameters import param, make_parser, choice, parse_choice
from tools import Struct

import detection.models as models

train_parameters = Struct (
    lr              = param(1.0,    help='learning rate'),
    lr_epoch_decay  = param(0.1,    help='decay lr during epoch by factor'),

    fine_tuning     = param(0.1,    help='fine tuning as proportion of learning rate'),

    momentum        = param(0.5,    help='SGD momentum'),
    seed            = param(1,      help='random seed'),
    batch_size      = param(4,     help='input batch size for training'),
    epoch_size      = param(1024,   help='epoch size for training'),

    num_workers     = param(4,      help='number of workers used to process dataset'),
    model           = choice(default='fcn', options=models.parameters, help='model type and parameters e.g. "fcn --start=4"')
)



detection_parameters = Struct (
    min_scale   = param(2/3,   help='minimum scaling during preprocessing'),
    max_scale   = param(3/2,   help='maximum scaling during preprocessing'),
    gamma       = param(0.15,  help='variation in gamma (brightness) when training'),
    image_size  = param(440,   help='size of patches to train on'),
    no_crop     = param(False, help='train always on full size images rather than cropping'),
    down_scale  = param(1,     help='down scale of image_size to test/train on'),
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
