
from scripts.history import history_summary, extract_histories, action_histogram, image_summaries, running_mAP
from scripts.datasets import load_dataset, annotation_summary, set_category_all, get_category, set_category

from os import path
import torch

from tools import struct, to_structs, filter_none, drop_while, concat_lists, map_dict, pprint_struct, pluck_struct

base_path = '/home/oliver/storage/export/'

penguins_dad = struct(
    hallett = 'dad/penguins_hallett.json',
    cotter = 'dad/penguins_cotter.json',
    royds = 'dad/penguins_royds.json',
)

penguins_new = struct(
    hallett = 'new/penguins_hallett.json',
    cotter = 'new/penguins_cotter.json',
    royds = 'new/penguins_royds.json',
)


def load_all(datasets, base_path):

    def load(filename):
        return load_dataset(path.join(base_path, filename))

    return datasets._map(load)
    # pprint_struct(pluck_struct('s


def combine_penguins(datasets):
    val_royd = set_category_all(get_category(datasets.royd, 'validate'), 'test_royd')
    val_hallett = set_category_all(get_category(datasets.hallett, 'validate'), 'test_hallett')
    val_cotter = set_category_all(get_category(datasets.cotter, 'validate'), 'test_cotter')

    train = list(concat_lists([get_category(datasets.royd, 'train'), 
        get_category(datasets.cotter, 'train'), 
        get_category(datasets.hallett, 'train')]))   

    combined = struct(config=datasets.royd.config, images=concat_lists([train, val_royd, val_hallett, val_cotter]))

    with open(combined_file, 'w') as outfile:
        json.dump(to_dicts(combined), outfile, sort_keys=True, indent=4, separators=(',', ': '))




if __name__ == '__main__':
    loaded = load_all(other, base_path)