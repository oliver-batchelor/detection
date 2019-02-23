from scripts.datasets import load_dataset, get_counts
from os import path

from tools import struct, to_structs, filter_none, drop_while, concat_lists, map_dict, pprint_struct, pluck_struct




base_path = '/home/oliver/storage/export/'


def load(filename):
    dataset = load_dataset(path.join(base_path, filename))

    counts = get_counts(dataset)


datasets = struct(
    scott_base = 'scott_base.json',
)

if __name__ == '__main__':
         

    loaded = datasets._map(load)

    # plot_counts(path.join())