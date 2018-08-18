
import arguments
from dataset.annotate import load_dataset
from detection.display import display_batch

from arguments import detection_parameters
from tools.parameters import parse_args

from tools import Struct
from tools.parameters import param, required, parse_args
from tools.image import cv
import pprint

pp = pprint.PrettyPrinter(indent=2)


test_parameters = Struct (
    batch_size  = param(1,           help='batch size to display'),
    test        = param(False,       help='show test images instead of training'),
    input       = required('str',     help='json to load dataset'),
    num_workers = param(1,           help='number of dataloader workers'),
    epoch_size  = param(1024,        help='number of dataloader workers'),
)

parameters = detection_parameters.merge(test_parameters)
args = parse_args(parameters, 'display dataset images')

pp.pprint(args.to_dicts())

config, dataset = load_dataset(args.input)

def identity(batch):
    return batch

iter = dataset.test(args, collate_fn=identity) if args.test else dataset.sample_train(args, collate_fn=identity)


for i in iter:
    key = cv.display(display_batch(i, classes=dataset.classes))
    if(key == 27):
        break
