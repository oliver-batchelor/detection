
import argparse

from tools.parameters import param, parse_args, choice, parse_choice, group
from tools import struct

import detection.models as models

train_parameters = struct (
    optimizer = group('optimizer settings',
        lr              = param(0.001,    help='learning rate'),

        lr_decay        = param('log', help='type of LR decay to use (log|cosine|step)'),
       
        lr_schedule     = param(30, type='int', help='epochs before dropping LR for step decay'),
        lr_step         = param(10, type='float', help='step factor to drop LR'),

        lr_min          = param(0.1,    help='minimum learning rate to decay to (factor of initial lr)'),
      
        fine_tuning     = param(1.0,    help='fine tuning as proportion of learning rate'),
        momentum        = param(0.5,    help='SGD momentum'),
        weight_decay    = param(1e-4, help='weight decay rate')
    ),

    average_start  = param(2,    help='start weight averaging after epochs'),
    average_window = param(1,    help='use a window of size for averaging, 1 = no averaging'),

    seed            = param(1,      help='random seed'),
    batch_size      = param(8,     help='input batch size for training'),

    reviews      = param(0,     help = 'number of reviews conducted per epoch'),
    detections   = param(0,     help = 'number of detections conducted per epoch on new images'),

    detect_all   = param(False,     help = 'run detections for all images'),
    variation_window = param(2,         help = 'size of window to compute frame variation with'),


    tests = param('test', help='comma separated list of test sets to use'),
    
    epoch_size          = param(1024, help='epoch size for training'),
    validation_pause    = param(16,    type='int', help='automatically pause training if validation does not improve after epochs'),

    incremental       = param(False, help='simulate incremental adding to the dataset during training'),
    max_epochs      = param(None, type='int', help='maximum number of epochs to train'),
      
    pause_epochs      = param(128, type='int', help='number of epochs to train before pausing'),

    eval_split      = param(False, help='evaluate images split into crops of image_size'),
    overlap         = param(200, type='int', help='margin of overlap when splitting images for evaluation'),
    
    box_noise       = param(0.0, help='add gaussian noise to bounding boxes'),
    box_offset      = param(0, help='add systematic offset to bounding boxes'),

    paused          = param(False, help='start trainer paused'),
    num_workers     = param(4,      help='number of workers used to process dataset'),
    model           = choice(default='fcn', options=models.parameters, help='model type and parameters e.g. "fcn --start=4"'),

    bn_momentum    = param(0.9, "momentum for batch normalisation modules"),

    no_load         = param(False,   help="don't attempt to load previously saved model"),
    dry_run         = param(False,   help="run for testing only (don't store results or log progress)"),

    restore_best    = param(False,   help="restore weights from best validation model"),

    log_dir         = param(None, type='str', help="output directory for logging"),
    run_name        = param('training', help='name for training run')
)


detection_parameters = struct (
    image = group('image',
        gamma           = param(0.15,  help='variation in gamma when training'),
        channel_gamma   = param(0.0,  help='variation per channel gamma when training'),

        brightness      = param(0.05,  help='variation in brightness (additive) when training'),
        contrast        = param(0.05,  help='variation in contrast (multiplicative) when training'),

        hue             = param(0.05,  help='variation in brightness (additive) when training'),
        saturation      = param(0.00,  help='variation in contrast (multiplicative) when training'),        

        image_size      = param(600,   help='size of patches to train on'),

        border_bias     = param(0.1,    help = "bias random crop to select border more often (proportion of size)"),
        augment = param("crop", help = 'image augmentation method (crop | full | ssd)'),

        scale  = param(1.0,     help='base scale of image_size'),
        resize = param(None, type='float', help='resize short side of images to this dimension'),

        max_scale = param(4./3, help='maximum scale multiplier for cropped patches'),
        min_scale = param(None, type='float', help = 'minimum scale multiplier for cropped patches (default 1/max_scale)'),

        max_aspect = param(1.1, help = 'maximum aspect ratio on crops'),

        transposes  = param(False, help='enable image transposes in training'),
        flips          = param(True, help='enable horizontal image flips in training'),
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

    select_instance = param(0.5, help = 'probability of cropping around an object instance as opposed to a random patch'),
    min_visible     = param (0.4, help = 'minimum proportion of area for an overlapped box to be included'),
    crop_boxes      = param(False, help='crop boxes to the edge of the image patch in training'),
    
    top_anchors     = param(1,     help='select n top anchors for ground truth regardless of iou overlap'),
    overlap_attenuation = param(False, help='weight anchor box targets by the amount of overlap iou'),
)


debug_parameters =  struct (
    debug = group('debug',
        debug_predictions = param (False, "distribution of predictions by class"),
        debug_boxes       = param (False, "debug anchor box matches"),
        debug_all         = param (True, "enable all debug options")
    )
)


input_choices = struct(
    json = struct(
        path          = param(type="str", help = "path to exported json annotation file", required=True)),
    coco = struct(
        path          = param("/home/oliver/storage/coco", help = "path to exported json annotation file")),
    voc = struct(
        path          = param("/home/oliver/storage/voc", help = "path to exported json annotation file"),
        preset        = param("test2007", help='preset configuration of testing/training set used options test2007|val2012')
    )
)

def make_input_parameters(default = None, choices = input_choices):
    return struct(
        keep_classes = param(type="str", help = "further filter the classes, but keep empty images"),
        subset       = param(type="str", help = "use a subset of loaded classes, filter images with no anntations"),
        input        = choice(default=default, options=choices, help='input method', required=default is None),
    )


input_remote = make_input_parameters('remote', input_choices._extend(
    remote = struct (host = param("localhost:2160", help = "hostname of remote connection"))
))

parameters = detection_parameters._merge(train_parameters)._merge(input_remote)._merge(debug_parameters)


def get_arguments():
    args = parse_args(parameters, "trainer", "object detection parameters")


    args.model = parse_choice("model", parameters.model, args.model)
    args.input = parse_choice("input", parameters.input, args.input)


    return args
