from os import path
import numpy as np
import cv2

import torch

from tools import struct
from tools.parameters import param, parse_args

from tools.image import cv

from tools.dataset import flat


def load_image_with_depth(filename):
    base, ext = path.splitext(filename)

    depth = cv.imread_depth(base + ".depth")
    image = cv.imread_color(filename)

    return image, depth


def image_with_depth(filename):
    base, ext = path.splitext(filename)
    depth = base + ".depth"
    
    if(flat.image_file(filename) and path.exists(depth)):
        return filename


def display_image_depth(image, depth):
    depth.add_(-depth.min())
    depth = depth.div_(1000/256).clamp(max = 255).byte()

    depth = cv.gray_to_rgb(depth)

    cv.display(torch.cat([image, depth], 1))


parameters = struct (
    input = param('',    required = True,   help = "input folder")
)

args = parse_args(parameters, "display depth", "")
image_files = flat.find_files(args.input, image_with_depth)

for file in image_files:
    print(file)
    image, depth = load_image_with_depth(file)
    
    display_image_depth(image, depth)
