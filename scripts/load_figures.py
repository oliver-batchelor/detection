from scripts.figures import make_chart

from scripts.history import history_summary, extract_histories, \
     image_summaries, image_summary, correction_types, action_types

from scripts.datasets import load_dataset, annotation_summary

from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection

import tools.window as window


from os import path
import torch
import math

from tools import struct, to_structs, filter_none, drop_while, concat_lists, \
        map_dict, pprint_struct, pluck_struct, count_dict, sum_list, Struct, sum_dicts

import matplotlib.pyplot as plt
import csv

make_chart = make_chart

def load_all(datasets, base_path):

    def load(filename):
        print("loading: ", filename)

        dataset = load_dataset(path.join(base_path, filename))

        dataset.images = [image for image in dataset.images 
            if (image.category == 'train' or image.category == "validate")]

        summary = annotation_summary(dataset)
 
        history = extract_histories(dataset) 
        image_summaries, history_summaries = history_summary(history)

        summary = summary._merge(history_summaries)
        return struct (summary = summary, history = history, images = dataset.images, image_summaries=image_summaries)

    return datasets._map(load)
 
def export_csv(file, fields, rows):
    with open(file, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row._to_dicts())
    

base_path = '/home/oliver/storage/export/'

datasets = struct(
    penguins = 'penguins.json',
    branches = 'branches.json',
    seals1 = 'seals.json',
    seals2 = 'seals_shanelle.json',
    scott_base = 'scott_base.json',
    apples1 = 'apples.json',
    apples2 = 'apples_lincoln.json',
    scallops = 'scallops_niwa.json',  
    fisheye = 'victor.json',
    buoys       = 'mum/buoys.json',
    aerial_penguins = 'oliver/combined.json'
)
 
dataset_labels = struct(
    penguins = "penguins",
    branches = "branches",
    seals1 = "seals",
    seals2 = "$seals_b$",

    scott_base = "scott base",
    apples1 = "$apples^1$",
    apples2 = "$apples^2$",

    scallops = "$scallops_e$",
    fisheye = "fisheye",
    buoys = "$buoys_d$",

    aerial_penguins = "penguin survey"
)

penguin_labels = struct(
    hallett_a = "hallett",
    cotter_a = 'cotter',
    royds_a = 'royds',
    hallett_b = "$hallett_c$",
    cotter_b = '$cotter_c$',
    royds_b = '$royds_c$',
)


penguins_a = struct(
    hallett_a = 'oliver/penguins_hallett.json',
    cotter_a = 'oliver/penguins_cotter.json',
    royds_a = 'oliver/penguins_royds.json'
)

penguins_b = struct(
    hallett_b = 'dad/penguins_hallett.json',
    cotter_b = 'dad/penguins_cotter.json',
    royds_b = 'dad/penguins_royds.json',
)


tab20 = plt.get_cmap("tab20")
paired = plt.get_cmap("Paired")
set1 = plt.get_cmap("Set1")

dataset_keys = list(datasets.keys()) + ["pascal_voc", "coco"]

dataset_colors = {k : tab20(i) for i, k in enumerate (dataset_keys) }
correction_colors = {k : paired(i) for i, k in enumerate (correction_types) }
action_colors = {k : set1(i) for i, k in enumerate (action_types) }

penguin_keys = [val for pair in \
    zip(sorted(penguins_b.keys()), sorted(penguins_a.keys())) for val in pair]


penguin_colors = {k : paired(i) for i, k in enumerate (penguin_keys) }

