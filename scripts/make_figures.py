import scripts.history
from scripts.datasets import load_dataset, annotation_summary

from os import path

from tools import struct, to_structs, filter_none, drop_while, concat_lists, map_dict, pprint_struct, pluck_struct


def load_all(datasets, base_path):

    def load(filename):
        dataset = load_dataset(path.join(base_path, filename))
        summary = annotation_summary(dataset)

        return struct (summary = summary)

    loaded = datasets._map(load)

    pprint_struct(pluck_struct('summary', loaded))

    
    
datasets = struct(
    penguins = 'penguins.json',
    # scallops = 'scallops.json',
    trees   = 'trees_josh.json',
    #hallett = 'penguins_hallett.json',
    #cotter = 'penguins_cotter.json',
    #royds = 'penguins_royds.json',
    combined = 'penguins_combined.json',
)

base_path = '/home/oliver/storage/export/'


if __name__ == '__main__':
    load_all(datasets, base_path)