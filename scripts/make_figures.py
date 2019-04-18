from scripts.history import history_summary, extract_histories, \
     image_summaries, running_mAP, image_summary, correction_types, action_types

from scripts.datasets import load_dataset, annotation_summary
from scripts.figures import *

from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection

import tools.window as window


from os import path
import torch
import math

from tools import struct, to_structs, filter_none, drop_while, concat_lists, \
        map_dict, pprint_struct, pluck_struct, count_dict, sum_list, Struct, sum_dicts



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
 
     
def plot_instances_time(dataset, smoothing = 1):

    summaries = image_summaries(dataset.history)

    instances = torch.Tensor(pluck('instances', summaries))
    durations = torch.Tensor(pluck('duration', summaries))
     
    fig, ax = plt.subplots(figsize=(24, 12))

    plt.plot(range(instances.size(0)), (instances / durations).numpy(), 'x')

    smoothed = (window.rolling_mean(instances, window=11) / window.rolling_mean(durations, window=11)).numpy()
    plt.plot(range(instances.size(0)), smoothed)

    plt.show()


def running_mAPs(datasets, window=250, iou=0.5):

    fig, ax = plt.subplots(figsize=(24, 12))

    for k, dataset in datasets.items():
        summaries = image_summaries(dataset.history)
        durations = torch.Tensor(pluck('duration', summaries)).cumsum(0)

        scores = running_mAP(dataset.history, window=window, iou=iou)
        plt.plot((durations / 60).numpy(), scores, label = k)
        # plt.plot(range(len(scores)), scores, label = k)

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)

    ax.set_title("Running mAP vs Anotation time")

    ax.set_xlabel("Time (m)")
    ax.set_ylabel("mAP @" + iou)

    ax.legend()
    plt.show()



        

def cumulative_instances(datasets):

    fig, ax = plt.subplots(figsize=(24, 12))

    for k in sorted(datasets.keys()):
        dataset = datasets[k]
        
        summaries = image_summaries(dataset.history)

        instances = torch.Tensor(pluck('instances', summaries)).cumsum(0)
        durations = torch.Tensor(pluck('duration', summaries)).cumsum(0)

        plt.plot((durations / 60).numpy(), instances.numpy(), label = k)

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)

    ax.set_title("Cumulative annotated instances vs. time")

    ax.set_xlabel("Annotation time (m)")
    ax.set_ylabel("Count")

    ax.set_title("Cumulative instances vs Annotation time")

    ax.legend()
    
    return fig, ax


# def summary_figures(datasets):


    
base_path = '/home/oliver/storage/export/'


datasets = struct(
    penguins = 'penguins.json',
    branches = 'branches.json',

    seals = 'seals.json',
    scott_base = 'scott_base.json',
    apples1 = 'apples.json',
    apples2 = 'apples_lincoln.json',

    scallops = 'scallops_niwa.json', 
  
    fisheye = 'victor.json',
    buoys       = 'mum/buoys.json',

    aerial_penguins = 'oliver/combined.json',
)


penguins_a = struct(
    hallett_a = 'oliver/penguins_hallett.json',
    cotter_a = 'oliver/penguins_cotter.json',
    royds_a = 'oliver/penguins_royds.json',
)

penguins_b = struct(
    hallett_b = 'dad/penguins_hallett.json',
    cotter_b = 'dad/penguins_cotter.json',
    royds_b = 'dad/penguins_royds.json',
)


tab10 = plt.get_cmap("tab10")
paired = plt.get_cmap("Paired")
set1 = plt.get_cmap("Set1")

dataset_colors = {k : tab10(i) for i, k in enumerate (datasets.keys()) }
correction_colors = {k : paired(i) for i, k in enumerate (correction_types) }
action_colors = {k : set1(i) for i, k in enumerate (action_types) }

penguin_keys = [val for pair in \
    zip(sorted(penguins_b.keys()), sorted(penguins_a.keys())) for val in pair]


penguin_colors = {k : paired(i) for i, k in enumerate (penguin_keys) }


if __name__ == '__main__':
    loaded = load_all(datasets, base_path)
    pprint_struct(pluck_struct('summary', loaded))


    #confidence_iou_scatter(loaded)
    # annotation_histogram(loaded.penguins, n_splits=10)

    loaded = load_all(penguins_all, base_path)
    pprint_struct(pluck_struct('summary', loaded))

    # oliver = load_all(penguins_oliver, base_path)
    # dad = load_all(penguins_dad, base_path)

    cumulative_instances(loaded)

    # loaded = load_all(penguins, base_path)
    # pprint_struct(pluck_struct('summary', loaded))


    # loaded = load_all(other, base_path)
    # pprint_struct(pluck_struct('summary', loaded))

    # actions_time(loaded)
    # plot_actions(loaded.apples)



    # cumulative_instances(loaded)
    # running_mAPs(loaded, iou=0.75)


    # load_all(penguins_oliver, base_path)
    # load_all(penguins_dad, base_path)