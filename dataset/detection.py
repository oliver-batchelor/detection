from os import path
import random
import math

import torch
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader, default_collate


import tools.dataset.direct as direct
from tools import transpose, over

from tools.dataset.flat import FlatList
from tools.dataset.samplers import RepeatSampler
from tools.image import transforms, cv

from tools.image.index_map import default_map
from tools import tensor

from detection import box

def load_boxes(image):
    #print(image)
    img = cv.imread_color(image['file'])
    return {**image, 'image':img}


def random_mean(mean, magnitude):
    return mean + random.uniform(-magnitude, magnitude)


def scale(scale):
    def apply(d):
        image, boxes = d['image'], d['boxes']

        boxes = box.transform(boxes, (0, 0), (scale, scale))
        return {**d,
                'image': transforms.resize_scale(image, scale),
                'boxes': boxes
            }
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
        image, input_labels, input_boxes = d['image'], d['labels'], d['boxes']

        if transpose:
            image = image.transpose(0, 1)
            input_boxes = box.transpose(input_boxes)

        input_size = (image.size(1), image.size(0))
        region_size = (cw / sx, ch / sy)

        # x, y = transforms.random_region(input_size, region_size, border)

        boxes = input_boxes.new()
        labels = input_labels.new()

        sx = sx * (-1 if flip else 1)
        sy = sy * (-1 if vertical_flip else 1)

        # while boxes.size(0) == 0 and input_boxes.size(0) > 0:
        x, y = transforms.random_region(input_size, region_size, border)

        x_start = x + (region_size[0] if flip else 0)
        y_start = y + (region_size[1] if vertical_flip else 0)

        if input_boxes.size(0) > 0:
            boxes = box.transform(input_boxes, (-x_start, -y_start), (sx, sy))
            boxes, labels = box.filter_hidden(boxes, input_labels, (0, 0), dest_size, min_visible=min_visible)

        if crop_boxes:
            box.clamp(boxes, (0, 0), dest_size)


        centre = (x + region_size[0] * 0.5, y + region_size[1] * 0.5)
        t = transforms.make_affine(dest_size, centre, scale=(sx, sy))

        return {**d,
                'image': transforms.warp_affine(image, t, dest_size),
                'boxes': boxes,
                'labels': labels
            }
    return apply




def load_training(args, dataset, collate_fn=default_collate):
    n = round(args.epoch_size / args.image_samples)
    return DataLoader(dataset,
        num_workers=args.num_workers,
        batch_size=1 if args.full_size else args.batch_size,
        sampler=RepeatSampler(n, len(dataset)) if args.epoch_size else RandomSampler(dataset),
        collate_fn=collate_fn)


def sample_training(args, images, loader, transform, collate_fn=default_collate):

    dataset = direct.Loader(loader, transform)
    sampler = direct.RandomSampler(images, (args.epoch_size // args.image_samples)) if args.epoch_size else direct.ListSampler(images)

    return DataLoader(dataset,
        num_workers=args.num_workers,
        batch_size=1 if args.full_size else (args.batch_size // args.image_samples),
        sampler=sampler,
        collate_fn=collate_fn)


def load_testing(args, images, collate_fn=default_collate):
    return DataLoader(images, num_workers=args.num_workers, batch_size=1, collate_fn=collate_fn)


def encode_targets(encoder, crop_boxes=False):
    def f(d):
        image = d['image']
        targets = encoder.encode(image, d['boxes'], d['labels'], crop_boxes=crop_boxes)

        return {
            'image':image,
            'targets': targets,
            'lengths': len(d['labels'])
        }
    return f

def identity(x):
    return x

def transform_training(args, encoder=None):
    s = 1 / args.down_scale
    result_size = int(args.image_size * s)

    crop = random_crop((result_size, result_size), scale_range = (s * args.min_scale, s * args.max_scale),
        non_uniform_scale = 0.1, flips=args.flips, transposes=args.transposes, vertical_flips=args.vertical_flips,
        crop_boxes=args.crop_boxes, allow_empty=args.allow_empty)

    adjust_colors = over('image', transforms.adjust_gamma(args.gamma, args.channel_gamma))

    encode = identity if encoder is None else  encode_targets(encoder, args.crop_boxes)
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


    def train(self, args, encoder=None, collate_fn=default_collate):
        images = FlatList(list(self.train_images.values()), loader = load_boxes,
            transform = transform_training(args, encoder=encoder))

        return load_training(args, images, collate_fn=flatten(collate_fn))

    def sample_train(self, args, encoder=None, collate_fn=default_collate):
        return sample_training(args, list(self.train_images.values()), load_boxes,
            transform = transform_training(args, encoder=encoder), collate_fn=flatten(collate_fn))

    def load_testing(self, file, args):
        transform = transform_image_testing(args)
        image = cv.imread_color(file)

        if transform is not None:
            image = transform(image)
        return image

    def test(self, args, collate_fn=default_collate):
        images = FlatList(list(self.test_images.values()), loader = load_boxes, transform = transform_testing(args))
        return load_testing(args, images, collate_fn=collate_fn)

    def test_training(self, args, collate_fn=default_collate):
        images = FlatList(list(self.train_images.values()), loader = load_boxes, transform = transform_testing(args))
        return load_testing(args, images, collate_fn=collate_fn)
