
import json
from dataset import annotate

import argparse

from tools import struct, to_structs
from detection import evaluate



def decode_image(data, config):
    image = annotate.decode_image(data, config=config)
    history = list(reversed(data.history))

    

    # tags = [(entry.tag) for time, entry in data.history]
    
    time, edit = history[0]
    detections = annotate.decode_detections(edit.contents.contents, annotate.class_mapping(config))
    

    test = struct(prediction = detections._sort_on('confidence'), target = image.target)
    compute_mAP = evaluate.mAP_classes([test], num_classes = len(config.classes))

    pr = compute_mAP(0.5)
    print(image.file, pr.total.mAP)

    return image._extend(detections = detections)


def decode_dataset(data):
    data = to_structs(data)

    config = data.config
    classes = [struct(id = int(k), name = v) for k, v in config.classes.items()]

    images = [decode_image(i, config) for i in data.images if len(i.history) > 0]


def load_dataset(filename):
    with open(filename, "r") as file:
        str = file.read()
        return decode_dataset(json.loads(str))
    raise Exception('load_file: file not readable ' + filename)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process history dump.')
    parser.add_argument('--input', type=str, help = 'input json file')

    args = parser.parse_args()

    load_dataset(args.input)