
import argparse
import tools.arguments as common

from tools.parameters import param, make_parser
from tools import Struct



import models

image_parameters = Struct (
    min_scale   = param(2/3,   help='minimum scaling during preprocessing'),
    max_scale   = param(3/2,   help='maximum scaling during preprocessing'),
    gamma       = param(0.15,  help='variation in gamma (brightness) when training'),
    image_size  = param(440,   help='size of patches to train on'),
    no_crop     = param(False, help='train always on full size images rather than cropping'),
    down_scale  = param(1,     help='down scale of image_size to test/train on')
)



parameters = image_parameters.merge(common.parameters)

parser = make_parser('Object detection', parameters)
parser.add_argument('--input', required=True, help='input image path')
parser.add_argument('--model', action='append', default=[], help='model type and sub-parameters e.g. "unet --dropout 0.1"')
parser.add_argument('--dataset', default='annotate', help='dataset type options are (annotate)')


def get_arguments():
    return parser.parse_args()

def default_arguments():
    return parser.parse_args([])
