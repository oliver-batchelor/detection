from scripts.history import history_summary, extract_histories
from scripts.datasets import load_dataset, annotation_summary

from os import path

from tools import struct, to_structs, filter_none, drop_while, concat_lists, map_dict, pprint_struct, pluck_struct


def load_all(datasets, base_path):

    def load(filename):
        dataset = load_dataset(path.join(base_path, filename))

        dataset.images = [image for image in dataset.images if len(image.history) > 0 
            and (image.category == 'train' or image.category == "validate")]

        summary = annotation_summary(dataset)
        
        history = extract_histories(dataset) 
        summary = summary._merge(history_summary(history))


        return struct (summary = summary, history = history)

    loaded = datasets._map(load)

    pprint_struct(pluck_struct('summary', loaded))

    
    
datasets = struct(
    penguins = 'penguins.json',
    # trees_josh = 'trees_josh.json',
    # scallops = 'scallops.json',
    # hallett = 'penguins_hallett.json',
    # cotter = 'penguins_cotter.json',
    # royds = 'penguins_royds.json',
    # combined = 'penguins_combined.json',
)

base_path = '/home/oliver/storage/export/'


if __name__ == '__main__':
    load_all(datasets, base_path)