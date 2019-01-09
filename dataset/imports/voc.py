import argparse
import inspect
import os.path as path
import os

import torch
import shutil

from torch.utils.data import DataLoader

from tools.image import index_map, cv
from tools import tensor, struct

from xml.dom import minidom
from tools.image.index_map import default_colors

import xmltodict

train_val = [('train', 'VOC2012', 'trainval'), ('train', 'VOC2007', 'trainval')]

presets = struct (
    test2007 = train_val + [('test', 'VOC2007', 'test')],
    val2012 = [
        ('train', 'VOC2007', 'trainval'), ('train', 'VOC2007', 'test'), 
        ('train', 'VOC2012', 'train'), 
        ('test', 'VOC2012', 'val')]
)


def tagged(tag, contents):
    return {'tag':tag, 'contents':contents}

imagesets = struct (
    train="train.txt",
    val="val.txt",
    trainval="trainval.txt",
    test="test.txt")


def read_lines(file):
    with open(file) as g:
        return g.read().splitlines()




voc_classes = [ 'aeroplane', 'bicycle',  'bird',     'boat',
                'bottle',    'bus',      'car',      'cat',
                'chair',     'cow',      'diningtable', 'dog',
                'horse',     'motorbike', 'person',  'pottedplant',
                'sheep',     'sofa',     'train',    'tvmonitor']


class_map = {name: i for i, name in enumerate(voc_classes)}

def lookup_class(name):
    assert name in class_map, "not a valid VOC class " + name
    return class_map[name]


def map_list(f, xs):
    return list(map(f, xs))


def import_subset(input_path, year, subset, target_category, class_map):
    print("{}: importing {} from {}".format(target_category, subset, year))
    assert subset in imagesets


    def import_object(ann):
        class_name = ann['name']

        b = ann['bndbox']
        lower = [float(b['xmin']), float(b['ymin'])]
        upper = [float(b['xmax']), float(b['ymax'])]

        return {
            'label': class_map[class_name],
            'confirm': True,
            'detection': None,
            'shape': tagged('box', {'lower': lower, 'upper': upper })
        }

    def import_image(root):
        objects = root['object']

        if not type(root['object']) == list:
            objects = [root['object']]

        annotations = map_list(import_object, objects)
        file_name = path.join(year, 'JPEGImages', root['filename'])

        size = root['size']
        return {
            'imageFile':file_name,
            'imageSize':[size['width'], size['height']],
            'category':target_category,
            'annotations':annotations
        }


    
    imageset_file = path.join(input_path, year, "ImageSets/Main", imagesets[subset])
    image_ids = read_lines(imageset_file)

    images = []

    for i in image_ids:
        annotation_path = path.join(input_path, year, "Annotations", i + ".xml")

        with open(annotation_path, "r") as g:
            xml = g.read()
            root = xmltodict.parse(xml)['annotation']
            images.append(import_image(root))

    return images


def make_dataset(root, images, class_list):
    classes = {}
    for i, class_name in enumerate(class_list):
        assert class_name in voc_classes, "not a VOC class: " + class_name
        classes[i] = {
            'name':class_name,
            'colour':default_colors[i],
            'shape':'box'
        }

    return {
        'config' : {
            'root':root,
            'extensions':[".jpg"],
            'classes':classes
        },
        'images' : images
    }
    

        


def import_images(input_path, preset):

    return [image 
        for category, year, subset in presets[preset]
        for image in import_subset(input_path, year, subset, category, class_map) 
    ]

def import_voc(input_path, preset):
    images = import_images(input_path, preset)
    return make_dataset(input_path, images, voc_classes)    



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pascal VOC, import dataset')

    parser.add_argument('--input', default='/home/oliver/storage/voc',
                        help='input image path')

    parser.add_argument('--output', default=None, required=True,
                        help='convert dataset and output to path')

    parser.add_argument('--image_classes', default=None,
                    help='use images containing classes (comma sep)')

    parser.add_argument('--restrict', default=None,
                    help='restrict instances to these classes (comma sep)')


    parser.add_argument('--preset', default=None, required=True,
                    help='preset configuration of testing/training set used options test2007|val2012')


    args = parser.parse_args()

 
    classes = args.restrict.split(",") if args.restrict else voc_classes
    image_classes = args.image_classes.split(",") if args.image_classes else classes
    
 
    images = import_images(args.input, args.preset)
    images = filter_images(images, map_list(lookup_class, classes), map_list(lookup_class, image_classes))

    dataset = make_dataset(args.input, images, classes)
    summarise(images, classes)

    with open(args.output, 'w') as outfile:
        json.dump(dataset, outfile, sort_keys=True, indent=4, separators=(',', ': '))