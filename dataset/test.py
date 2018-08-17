
import arguments
from dataset.annotate import load_dataset
from detection.display import display_batch

from arguments import detection_parameters
from tools.parameters import parse_args

from tools import Struct
from tools.parameters import param, parse_args
from tools.image import cv

args = arguments.get_arguments()


test_parameters = Struct (
    batch_size  = param(1,           help='batch size to display'),
    test        = param(False,       help='show test images instead of training'),
    input       = param(None,        help='json to load dataset', type='str'),
    num_workers = param(1,           help='number of dataloader workers'),
    epoch_size  = param(1024,        help='number of dataloader workers'),
)

args = parse_args(detection_parameters.merge(test_parameters), 'display dataset images')

assert args.input, "required argument --input"

config, dataset = load_dataset(args.input)

def identity(batch):
    return batch

iter = dataset.train(args, collate_fn=identity) if args.test else dataset.test(args, collate_fn=identity)

for i in iter:
    key = cv.display(display_batch(i, classes=dataset.classes))
    if(key == 27):
        break
