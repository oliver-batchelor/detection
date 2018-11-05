from os import path
import random
import math

import torch
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader, default_collate


import tools.dataset.direct as direct
from tools import transpose, over_struct

from tools.dataset.flat import FlatList
from tools.dataset.samplers import RepeatSampler
from tools.image import transforms, cv

from tools.image.index_map import default_map
from tools import tensor, Struct, Tensors

from detection import box
import collections


def collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])

    
    if isinstance(batch[0], Struct):
        d =  {key: collate([d[key] for d in batch]) for key in batch[0]}
        return Struct(**d)
    if isinstance(batch[0], Tensors):
        d =  {key: torch.cat([d[key] for d in batch]) for key in batch[0]}
        return Tensors(**d)        
    elif isinstance(batch[0], str):
        return batch        
    elif isinstance(batch[0], collections.abc.Mapping):
        return {key: collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    else:
        return default_collate(batch) 

    raise TypeError((error_msg.format(type(batch[0]))))





def load_boxes(image):
    img = cv.imread_color(image.file)
    return image.extend(image = img)


def random_mean(mean, magnitude):
    return mean + random.uniform(-magnitude, magnitude)


def scale(scale):
    def apply(d):
        image, target = d.image, d.target

        boxes = box.transform(target.boxes, (0, 0), (scale, scale))
        return d.extend(
                image   = transforms.resize_scale(image, scale),
                target = Struct(boxes = boxes, labels = target.labels))
    return apply

def random_log(l, u):
    return math.exp(random.uniform(math.log(l), math.log(u)))


def random_crop(dest_size, scale_range=(1, 1), non_uniform_scale=0, border = 0, min_visible=0.2, allow_empty=0.0, crop_boxes=False, flips=True, transposes=False, vertical_flips=False):
    cw, ch = dest_size

    def apply(d):

        scale = random_log(*scale_range)
        flip = flips and (random.uniform(0, 1) > 0.5)
        vertical_flip = vertical_flips and (random.uniform(0, 1) > 0.5)

        transpose = transposes and (random.uniform(0, 1) > 0.5)
        sx, sy = random_mean(1, non_uniform_scale) * scale, random_mean(1, non_uniform_scale) * scale
        image, input_target = d.image, d.target

        if transpose:
            image = image.transpose(0, 1)
            input_boxes = box.transpose(input_boxes)

        input_size = (image.size(1), image.size(0))
        region_size = (cw / sx, ch / sy)

        # x, y = transforms.random_region(input_size, region_size, border)
       # target = Struct(boxes = input_target.boxes.new(0, 4),  labels = input_target.labels.new(0))

        sx = sx * (-1 if flip else 1)
        sy = sy * (-1 if vertical_flip else 1)

        # while boxes.size(0) == 0 and input_boxes.size(0) > 0:
        x, y = transforms.random_region(input_size, region_size, border)

        x_start = x + (region_size[0] if flip else 0)
        y_start = y + (region_size[1] if vertical_flip else 0)


        target = input_target.extend(boxes = box.transform(input_target.boxes, (-x_start, -y_start), (sx, sy)))
        target = box.filter_hidden(target, (0, 0), dest_size, min_visible=min_visible)

        if crop_boxes:
            box.clamp(target.boxes, (0, 0), dest_size)


        centre = (x + region_size[0] * 0.5, y + region_size[1] * 0.5)
        t = transforms.make_affine(dest_size, centre, scale=(sx, sy))

        return d.extend(
                image = transforms.warp_affine(image, t, dest_size),
                target = target
            )
    return apply




def load_training(args, dataset, collate_fn=collate):
    n = round(args.epoch_size / args.image_samples)
    return DataLoader(dataset,
        num_workers=args.num_workers,
        batch_size=1 if args.full_size else args.batch_size,
        sampler=RepeatSampler(n, len(dataset)) if args.epoch_size else RandomSampler(dataset),
        collate_fn=collate_fn)


def sample_training(args, images, loader, transform, collate_fn=collate):

    dataset = direct.Loader(loader, transform)
    sampler = direct.RandomSampler(images, (args.epoch_size // args.image_samples)) if args.epoch_size else direct.ListSampler(images)

    return DataLoader(dataset,
        num_workers=args.num_workers,
        batch_size=1 if args.full_size else (args.batch_size // args.image_samples),
        sampler=sampler,
        collate_fn=collate_fn)


def load_testing(args, images, collate_fn=collate):
    return DataLoader(images, num_workers=args.num_workers, batch_size=1, collate_fn=collate_fn)


def encode_target(encoder, crop_boxes=False):
    def f(d):
        encoding = encoder.encode(d.image, d.target, crop_boxes=crop_boxes)

        return Struct(
            image   = d.image,
            encoding = encoding,
            lengths = len(d.target.labels)
        )
    return f

def identity(x):
    return x

def transform_training(args, encoder=None):
    s = 1 / args.down_scale
    result_size = int(args.image_size * s)

    crop = random_crop((result_size, result_size), scale_range = (s * args.min_scale, s * args.max_scale),
        non_uniform_scale = 0.1, flips=args.flips, transposes=args.transposes, vertical_flips=args.vertical_flips,
        crop_boxes=args.crop_boxes, allow_empty=args.allow_empty)

    adjust_colors = over_struct('image', transforms.adjust_gamma(args.gamma, args.channel_gamma))

    encode = identity if encoder is None else  encode_target(encoder, args.crop_boxes)
    return multiple(args.image_samples, transforms.compose (crop, adjust_colors, encode))

def multiple(n, transform):
    def f(data):
        return [transform(data) for _ in range(n)]
    return f

def flatten(collate_fn):
    def f(lists):
        return collate_fn([x for y in lists for x in y])
    return f

def transform_testing(args):
    if args.down_scale != 1:
        s = 1 / args.down_scale
        return scale(s)
    else:
        return None


def transform_image_testing(args):
    if args.down_scale != 1:
        return transforms.adjust_scale(1 / args.down_scale)
    else:
        return None


class DetectionDataset:

    def __init__(self, train_images={}, test_images={}, classes=[]):

        assert type(train_images) is dict, "expected train_images as a dict"
        assert type(test_images) is dict, "expected test_images as a dict"
        assert type(classes) is list, "expected classes as a list"

        self.train_images = train_images
        self.test_images = test_images

        self.classes = classes


    def update_image(self, file, image, category):
        if file in self.train_images:
            del self.train_images[file]
        if file in self.test_images:
            del self.test_images[file]

        if image is not None:
            if category == 'Test':
                self.test_images[file] = image
            elif category == 'Train':
                self.train_images[file] = image


    def train(self, args, encoder=None, collate_fn=collate):
        images = FlatList(list(self.train_images.values()), loader = load_boxes,
            transform = transform_training(args, encoder=encoder))

        return load_training(args, images, collate_fn=flatten(collate_fn))

    def sample_train(self, args, encoder=None, collate_fn=collate):
        return sample_training(args, list(self.train_images.values()), load_boxes,
            transform = transform_training(args, encoder=encoder), collate_fn=flatten(collate_fn))

    def load_testing(self, file, args):
        transform = transform_image_testing(args)
        image = cv.imread_color(file)

        if transform is not None:
            image = transform(image)
        return image

    def test(self, args, collate_fn=collate):
        images = FlatList(list(self.test_images.values()), loader = load_boxes, transform = transform_testing(args))
        return load_testing(args, images, collate_fn=collate_fn)

    def test_training(self, args, collate_fn=collate):
        images = FlatList(list(self.train_images.values()), loader = load_boxes, transform = transform_testing(args))
        return load_testing(args, images, collate_fn=collate_fn)
