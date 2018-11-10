import argparse
import inspect
import os.path as path
import os

import torch
import shutil

from torch.utils.data import DataLoader

from tools.image import index_map, cv
from tools import tensor, Struct

from xml.dom import minidom
from dataset.annotate import decode_dataset

from tools.image.index_map import default_colors

import xmltodict


imagesets = Struct (
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
                'horse',     'motorbike', 'person',  'potted plant',
                'sheep',     'sofa',     'train',    'tv/monitor']



def export_subset(input_path, year, subset, target_category, class_map):
    
    imageset_file = path.join(input_path, year, "ImageSets/Main", subset)
    image_ids = read_lines(imageset_file)

    images = []

    for i in image_ids:
        annotation_path = path.join(input_path, year, "Annotations", i + ".xml")

        with open(annotation_path, "r") as g:
            xml = g.read()
            root = xmltodict.parse(xml)['annotation']

            file_name = path.join(year, 'JPEGImages', root['filename'])


        def export_object(ann):
            class_name = ann['name']

            b = ann['bndbox']
            lower = [b['xmin'], b['ymin']]
            upper = [b['xmax'], b['ymax']]

            return {
              'label': class_map[class_name],
              'confirm': True,
              'detection': None,
              'shape': tagged('BoxShape', {'lower': lower, 'upper': upper })
            }

            objects = root['object'] if type(objects) == list else [root['object']]
            annotations = list(map(export_object, objects))


            size = root['size']
            image = {
                'imageFile':file_name,
                'imageSize':[size['width'], size['height']],
                'category':target_category,
                'annotations':annotations
            }

                
            images.append(image)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pascal VOC, export dataset')

    parser.add_argument('--input', default='/home/oliver/storage/voc',
                        help='input image path')

    parser.add_argument('--output', default=None, required=True,
                        help='convert dataset and output to path')

    parser.add_argument('--restrict', default=None,
                    help='restrict to classes (comma sep) when converting dataset')


    parser.add_argument('--config', default=None,
                    help='configuration of testing/training set used options test2007|test2012')


    args = parser.parse_args()
    classes = args.restrict.split(",") if args.restrict else voc_classes

    classes = {}
    class_map = {}

    for i, class_name in enumerate(classes):
        assert class_name in voc_classes, "not a VOC class: " + class_name
        classes[i] = {
            'name':class_name,
            'colour':default_colors[i],
            'shape':'BoxConfig'
        }

        class_map[class_name] = i
    

    export_subset(args.input, "VOC2012", imagesets.trainval, 'Train', class_map)

    # with open(args.output, 'w') as outfile:
    #     json.dump(all, outfile, sort_keys=True, indent=4, separators=(',', ': '))