from os import path
import random
import math

import torch
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.dataloader import DataLoader, default_collate


import tools.dataset.direct as direct

from tools.dataset.flat import FlatList
from tools.dataset.samplers import RepeatSampler
from tools.image import transforms, cv

from tools.image.index_map import default_map
from tools import over_struct, tensor, struct, table, cat_tables, Table, Struct, show_shapes


from detection import box
import collections


def collate_batch(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain Table, numbers, dicts or lists; found {}"

    elem = batch[0]
    elem_type = type(batch[0])

    if elem_type is Table:
        return cat_tables(batch)
           
    if elem_type is Struct:
        d =  {key: collate_batch([d[key] for d in batch]) for key in elem}
        return Struct(d)
    elif isinstance(elem, str):
        return batch        
    elif elem is None:
        return batch

    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        transposed = zip(*batch)
        return [collate_batch(samples) for samples in transposed]
    else:
        return default_collate(batch) 

    raise TypeError(error_msg.format(elem_type))



empty_target = table (
        bbox = torch.FloatTensor(0, 4),
        label = torch.LongTensor(0))


def load_image(image):
    img = cv.imread_color(image.file)
    return image._extend(image = img, image_size = torch.LongTensor([img.size(1), img.size(0)]))


def random_mean(mean, magnitude):
    return mean + random.uniform(-magnitude, magnitude)


def scale(scale):
    def apply(d):
        bbox = box.transform(d.target.bbox, (0, 0), (scale, scale))
        return d._extend(
                image   = transforms.resize_scale(d.image, scale),
                target = d.target._extend(bbox = bbox))
    return apply

def random_log(l, u):
    return math.exp(random.uniform(math.log(l), math.log(u)))

def random_flips(horizontal=True, vertical=False, transposes=False):
    def apply(d):
        image, bbox = d.image, d.target.bbox

        if transposes and (random.uniform(0, 1) > 0.5):
            image = image.transpose(0, 1)
            bbox = box.transpose(bbox)

        if vertical and (random.uniform(0, 1) > 0.5):
            image = cv.flip_vertical(image)
            bbox = box.flip_vertical(bbox, image.size(0))
       
        if horizontal and (random.uniform(0, 1) > 0.5):
            image = cv.flip_horizontal(image)
            bbox = box.flip_horizontal(bbox, image.size(1))
            
        return d._extend(image = image, target = d.target._extend(bbox = bbox))

    return apply


def resize_to(dest_size):
    cw, ch = dest_size

    def apply(d):      
        s = (cw / d.image.size(1), ch / d.image.size(0))

        return d._extend(
            image = transforms.resize_to(d.image, dest_size),
            target = d.target._extend(bbox = box.transform(d.target.bbox, scale = s))
        )

    return apply


def transformed(d, image, bbox):
    return d._extend(
        image   = image,
        target = d.target._extend(bbox = bbox))

# def transform(d, translate = (0, 0), scale = (1, 1)):

#     t = transforms.translation(dx, dy) 

#     return transformed(d, 
#         image = 
#         bbox = box.transform(d.target.bbox, translate = translate, scale = scale)
#     )


def centre_on(image_size):
    width, height = image_size

    def apply(d):
        dx = (width - d.image.size(1)) / 2
        dy = (height - d.image.size(0)) / 2

        bbox = box.transform(d.target.bbox, (dx, dy), (1, 1))
        image = transforms.warp_affine(d.image, transforms.translation(dx, dy), image_size)

        return transformed(d, image, bbox)
    return apply        


# def fit_to(image_size):
#     def apply(d):
#         h, w, _ = d.image.size()

#         s = image_size / max(h, w)
#         bbox = box.transform(d.target.bbox, scale = (s, s))

#       return d._extend(
#                 image   = transforms.warp_affine(d.image, transforms.translation(dx, dy), (image_size, image_size)),
#                 target = d.target._extend(bbox = bbox))

def as_tuple(bbox):
    b = bbox.tolist()
    return (b[0], b[1]), (b[2], b[3])

def random_crop_padded(dest_size, scale_range=(1, 1), aspect_range=(1, 1), border_bias=0, select_instance=0.5):
    cw, ch = dest_size

    def apply(d):

        scale = random_log(*scale_range)
        aspect = random_log(*aspect_range)

        sx, sy = scale * math.sqrt(aspect), scale / math.sqrt(aspect)

        input_size = (d.image.size(1), d.image.size(0))
        region_size = (cw / sx, ch / sy)

        num_instances = d.target.label.size(0)
        target_box = None
        
        x, y = transforms.random_crop_padded(input_size, region_size, border_bias=border_bias)

        if (random.uniform(0, 1) < select_instance) and num_instances > 0:
            instance = random.randint(0, num_instances - 1)
            x, y = transforms.random_crop_target(input_size, region_size, target_box=as_tuple(d.target.bbox[instance]))


        centre = (x + region_size[0] * 0.5, y + region_size[1] * 0.5)
        t = transforms.make_affine(dest_size, centre, scale=(sx, sy))

        return d._extend(
                image = transforms.warp_affine(d.image, t, dest_size, flags=cv.inter.cubic),
                target = d.target._extend(bbox = box.transform(d.target.bbox, (-x, -y), (sx, sy)))
            )
    return apply


# def ssd_crop():
    
#     options = [ None, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0 ]

#     def apply(d):
#         height, width, _ = d.image.size()

#         min_overlap = options[random.randint(0, len(options) - 1)]

#         # Return full image
#         if min_overlap is None:
#             return d

#         for i in range(0, 50):

#             w = random.randint(int(0.1 * width), width)
#             h = random.uniform(int(0.1 * height), height)

#             if max(w / h, h / w) > 2.0:
#                 continue

#             x = random.



#         return d._extend(
#                 image = transforms.warp_affine(d.image, t, dest_size),
#                 target = d.target._extend(bbox = box.transform(d.target.bbox, (-x, -y), (sx, sy)))
#             )
#     return apply


def filter_boxes(min_visible = 0.4, crop_boxes = False):
    
    def apply(d):
        size = (d.image.size(1), d.image.size(0))
        target = box.filter_hidden(d.target, (0, 0), size, min_visible=min_visible)

        if crop_boxes:
            box.clamp(target.bbox, (0, 0), size)    

        return d._extend(target = target)
    
    return apply


def load_training(args, dataset, collate_fn=collate_batch):
    n = round(args.epoch_size / args.image_samples)
    return DataLoader(dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        sampler=RepeatSampler(n, len(dataset)) if args.epoch_size else RandomSampler(dataset),
        collate_fn=collate_fn)


def sample_training(args, images, loader, transform, collate_fn=collate_batch):
    assert args.epoch_size is None or args.epoch_size > 0
    assert args.batch_size % args.image_samples == 0, "batch_size should be a multiple of image_samples"

    dataset = direct.Loader(loader, transform)
    sampler = direct.RandomSampler(images, (args.epoch_size // args.image_samples)) if (args.epoch_size is not None) else direct.ListSampler(images)

    return DataLoader(dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size // args.image_samples,
        sampler=sampler,
        collate_fn=collate_fn)


def load_testing(args, images, collate_fn=collate_batch):
    return DataLoader(images, num_workers=args.num_workers, batch_size=1, collate_fn=collate_fn)


def encode_target(encoder, crop_boxes=False, match_thresholds=(0.4, 0.5), match_nearest = 0):
    def f(d):
        encoding = encoder.encode(d.image, d.target, crop_boxes=crop_boxes, match_thresholds=match_thresholds, match_nearest = match_nearest)

        return struct(
            image   = d.image,
            encoding = encoding,
            target = d.target,
            lengths = len(d.target.label),
            id = d.id
        )
    return f

def identity(x):
    return x


def encode_with(args, encoder=None):
    return identity if encoder is None else  encode_target(encoder, crop_boxes=args.crop_boxes, 
        match_thresholds=(args.neg_match, args.pos_match), match_nearest = args.top_anchors)    


def transform_training(args, encoder=None):
    s = args.scale
    dest_size = (int(args.image_size * s), int(args.image_size * s))

    crop = identity

    if args.augment == "crop":
        crop = random_crop_padded(dest_size, scale_range = (s * 1/args.max_scale, s * args.max_scale), 
            aspect_range=(1/args.max_aspect, args.max_aspect), border_bias = args.border_bias, select_instance = args.select_instance)
    elif args.augment == "resize":
        crop = resize_to(dest_size)
    else:
        assert false, "unknown augmentation method " + args.augment

    filter = filter_boxes(min_visible=args.min_visible, crop_boxes=args.crop_boxes)
    flip   = random_flips(horizontal=args.flips, vertical=args.vertical_flips, transposes=args.transposes)
    
    
    adjust_light = over_struct('image', transforms.compose( 
        transforms.adjust_gamma(args.gamma, args.channel_gamma),
        transforms.adjust_brightness(args.brightness, args.contrast),
        transforms.adjust_colours(args.hue, args.saturation)
    ))

    encode = encode_with(args, encoder) 
    return multiple(args.image_samples, transforms.compose (crop, adjust_light, filter, flip, encode))

def multiple(n, transform):
    def f(data):
        return [transform(data) for _ in range(n)]
    return f

def flatten(collate_fn):
    def f(lists):
        return collate_fn([x for y in lists for x in y])
    return f


def transform_testing(args, encoder=None):
    """ Returns a function which transforms an image and ground truths for testing
    """
    s = args.scale
    dest_size = (int(args.image_size * s), int(args.image_size * s))

    transform = identity

    if args.augment == "crop":
        transform = scale(args.scale) if args.scale != 1 else identity      
        # return transforms.compose(scaling, centre_on(dest_size))

    elif args.augment == "resize":
        transform =  resize_to(dest_size)

    encode = encode_with(args, encoder)         
    return transforms.compose(transform, encode)


def least_recently_evaluated(images, n = None):
    random.shuffle(images)

    key = lambda image: tuple(image.evaluated or (0, 0))
    images = sorted(images, key=key)

    if n is not None:
        return images[:n]
    else:
        return images

class DetectionDataset:

    def __init__(self, images = {}, classes=[]):

        assert type(images) is dict, "expected images as a dict"
        assert type(classes) is list, "expected classes as a list"

        self.images = images
        self.classes = classes


    def update_image(self, image):
        self.images[image.id] = image

    def get_images(self, k):
        return [image for image in self.images.values() if image.category == k] 


    def mark_evalated(self, files, net_id):
        for k in files:
            assert k in self.images, "mark_evaluated, invalid file: " + k

            self.images[k].evaluated = net_id


    
    @property
    def train_images(self):
        return self.get_images('train')

    @property
    def test_images(self):
        return self.get_images('test')

    @property
    def validate_images(self):
        return self.get_images('validate')


    @property
    def new_images(self):
        return self.get_images('new')        

    @property
    def all_images(self):
        all_images = {}
        for k, images in self.images.items():
            all_images.update(images)

        return all_images

    def train(self, args, encoder, collate=collate_batch):
        images = FlatList(self.train_images, loader = load_image,
            transform = transform_training(args, encoder=encoder))

        return load_training(args, images, collate_fn=flatten(collate))

    def sample_train(self, args, encoder, collate=collate_batch):
        return sample_training(args, self.train_images, load_image,
            transform = transform_training(args, encoder=encoder), collate_fn=flatten(collate))

    def load_inference(self, id, file, args):
        transform = transform_testing(args)
        d = struct(id = id, file = file, target = empty_target)

        return transform(load_image(d)).image

    def test_on(self, images, args, encoder, collate=collate_batch):
        dataset = FlatList(images, loader = load_image, transform = transform_testing(args, encoder=encoder))
        return load_testing(args, dataset, collate_fn=collate)

    def test(self, args, encoder, collate=collate_batch):
        return self.test_on(self.test_images, args, encoder, collate=collate)

    def validate(self, args, encoder, collate=collate_batch):
        return self.test_on(self.validate_images, args, encoder, collate=collate)
