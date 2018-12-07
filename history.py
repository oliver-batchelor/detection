
import json
from dataset import annotate

import argparse

from tools import struct, to_structs

def decode_image(data, config):
    data.history 

    return annotate.decode_image(data, config=config)


def decode_dataset(data):
    data = to_structs(data)

    config = data.config
    classes = [struct(id = int(k), name = v) for k, v in config.classes.items()]

    images = [decode_image(i, config) for i in data.images if len(i.history) > 0]

    print(images)


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