
from os import path
import torch
import json

from scripts.datasets import load_dataset, set_category_all, get_category, set_category
from tools import struct, to_structs, filter_none, drop_while, concat_lists, map_dict, \
     pprint_struct, pluck_struct, pluck, to_dicts

base_path = '/home/oliver/storage/export/'

dad_files = struct(
    hallett = 'dad/penguins_hallett.json',
    cotter = 'dad/penguins_cotter.json',
    royds = 'dad/penguins_royds.json',
)

oliver_files = struct(
    hallett = 'oliver/penguins_hallett.json',
    cotter = 'oliver/penguins_cotter.json',
    royds = 'oliver/penguins_royds.json',
)

apples_files = struct(
   apples1 = 'apples.json',
   apples2 = 'apples_lincoln.json'
)


def load_all(datasets, base_path):

    def load(filename):
        return load_dataset(path.join(base_path, filename))

    return datasets._map(load)
    # pprint_struct(pluck_struct('s


def combine_penguins(datasets):
    val_royds = set_category_all(get_category(datasets.royds, 'validate'), 'test_royds')
    val_hallett = set_category_all(get_category(datasets.hallett, 'validate'), 'test_hallett')
    val_cotter = set_category_all(get_category(datasets.cotter, 'validate'), 'test_cotter')

    train = list(concat_lists([get_category(datasets.royds, 'train'), 
        get_category(datasets.cotter, 'train'), 
        get_category(datasets.hallett, 'train')]))   

    return struct(config=datasets.royds.config, images=concat_lists([train, val_royds, val_hallett, val_cotter]))

def combine_simple(datasets):
    images = concat_lists(pluck('images', datasets.values()))
    d = next(iter(datasets.values()))

    config = d.config._extend(root="/home/oliver/storage/penguins_combined")
    return struct(config=config, images=images)

  
def write_to(output, filename):

  with open(filename, 'w') as outfile:
        json.dump(to_dicts(output), outfile, sort_keys=True, indent=4, separators=(',', ': '))    



if __name__ == '__main__':
    dad = load_all(dad_files, base_path)
    oliver = load_all(oliver_files, base_path)

   # apples = load_all(apples_files, base_path)

    write_to(combine_simple(dad), path.join(base_path, "dad/combined.json"))
    write_to(combine_simple(oliver), path.join(base_path, "oliver/combined.json"))
    #write_to(combine_simple(apples), path.join(base_path, "apples_combined.json"))

    # write_to(combine_penguins(dad_files), path.join(base_path, "dad/combined_test.json"))
    # write_to(combine_penguins(oliver_files), path.join(base_path, "oliver/combined_test.json"))    