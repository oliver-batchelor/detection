from scripts.history import history_summary, extract_histories, action_histogram, image_summaries
from scripts.datasets import load_dataset, annotation_summary
from scripts.figures import *

import tools.window as window


from os import path
import torch

from tools import struct, to_structs, filter_none, drop_while, concat_lists, map_dict, pprint_struct, pluck_struct


def load_all(datasets, base_path):

    def load(filename):
        dataset = load_dataset(path.join(base_path, filename))

        dataset.images = [image for image in dataset.images 
            if (image.category == 'train' or image.category == "validate")]

        summary = annotation_summary(dataset)
        
        history = extract_histories(dataset) 
        summary = summary._merge(history_summary(history))

        return struct (summary = summary, history = history, images = dataset.images)

    return datasets._map(load)
    # pprint_struct(pluck_struct('summary', loaded))


# def combined_summary()
    
    
datasets = struct(
    # penguins = 'penguins.json',
    # trees_josh = 'trees_josh.json',
    scallops = 'scallops.json',
    # buoys = 'buoys.json',
    # branches = 'branches.json',

    # seals = 'seals.json',
    # scott_base = 'scott_base.json'


)

penguins_oliver = struct(
    hallett = 'oliver/penguins_hallett.json',
    cotter = 'oliver/penguins_cotter.json',
    # royds = 'oliver/penguins_royds.json',
)

penguins_dad = struct(
    hallett = 'dad/penguins_hallett.json',
    cotter = 'dad/penguins_cotter.json',
    royds = 'dad/penguins_royds.json',
)


def plot_actions(dataset, n_splits=None):
    keys = ['add', 'transform', 'confirm', 'submit', 'delete']
    
    actions = action_histogram(dataset.history, n_splits = n_splits)
    n = len(actions)
    print(n)

    fig, ax = plt.subplots(figsize=(24, 12))
    plot_stacks(np.array(range(n)) + 0.5, actions, keys, width= 0.5)

    plt.show()

     
def plot_instances_time(dataset, smoothing = 1):

    summaries = image_summaries(dataset.history)

    instances = torch.Tensor(pluck('instances', summaries))
    durations = torch.Tensor(pluck('duration', summaries))
     
    fig, ax = plt.subplots(figsize=(24, 12))

    plt.plot(range(instances.size(0)), (instances / durations).numpy(), 'x')

    smoothed = (window.rolling_mean(instances, window=11) / window.rolling_mean(durations, window=11)).numpy()
    plt.plot(range(instances.size(0)), smoothed)

    plt.show()


def cumulative_instances(dataset):

    summaries = image_summaries(dataset.history)

    instances = torch.Tensor(pluck('instances', summaries)).cumsum(0)
    durations = torch.Tensor(pluck('duration', summaries)).cumsum(0)

    fig, ax = plt.subplots(figsize=(24, 12))
    plt.plot(durations.numpy(), instances.numpy())

    plt.show()

    

base_path = '/home/oliver/storage/export/'


if __name__ == '__main__':
    loaded = load_all(datasets, base_path)
    
    oliver = load_all(penguins_oliver, base_path)
    # dad = load_all(penguins_dad, base_path)

    cumulative_instances(loaded.scallops)


    # pprint_struct(pluck_struct('summary', loaded))

    # load_all(penguins_oliver, base_path)
    # load_all(penguins_dad, base_path)