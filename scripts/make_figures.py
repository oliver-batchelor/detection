from scripts.history import history_summary, extract_histories, \
     image_summaries, running_mAP, image_summary

from scripts.datasets import load_dataset, annotation_summary
from scripts.figures import *

from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection

import tools.window as window


from os import path
import torch

from tools import struct, to_structs, filter_none, drop_while, concat_lists, \
        map_dict, pprint_struct, pluck_struct, count_dict


def load_all(datasets, base_path):

    def load(filename):
        print("loading: ", filename)

        dataset = load_dataset(path.join(base_path, filename))

        dataset.images = [image for image in dataset.images 
            if (image.category == 'train' or image.category == "validate")]

        summary = annotation_summary(dataset)
 
        history = extract_histories(dataset) 
        summary = summary._merge(history_summary(history))

        return struct (summary = summary, history = history, images = dataset.images)

    return datasets._map(load)
    # pprint_struct(pluck_struct('summary', loaded))

    
def actions_time(datasets):

    fig, ax = plt.subplots(figsize=(24, 12))
    scatters = []

    for k, dataset in datasets.items():
        summaries = image_summaries(dataset.history)

        instances = np.array(pluck('instances', summaries))
        duration = np.array(pluck('duration', summaries))
        actions = np.array(pluck('n_actions', summaries))

        plt.scatter(actions, duration, s=instances*5, alpha=0.5, label = k)

        # z = np.polyfit(actions, duration, 1)
        # p = np.poly1d(z)

        # plt.plot(actions, p(actions), '--')


    def update(handle, orig):
        handle.update_from(orig)
        handle.set_sizes([64])

    plt.legend(handler_map={PathCollection : HandlerPathCollection(update_func=update)})

    ax.set_ylim(ymin=0, ymax=150)
    ax.set_xlim(xmin=0, xmax=50)


    plt.show()



def image_histograms(dataset, get_histogram, keys, n_splits=None):
    actions = [get_histogram(image) for image in dataset.history]        

    if n_splits is not None:
        actions = sum_splits(actions, n_splits=n_splits)

    n = len(actions)
    
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_stacks(np.array(range(n)) + 0.5, actions, keys, width= 0.5)

    plt.show()    

def thresholds(image):
    changes = [action.value for action in image.actions if action.action=='threshold']
    return [image.threshold] + changes

def annotation_histogram(dataset, n_splits=None):
    def make_histogram(image):
        mapping = {'add':'false negative', 'confirm':'weak positive', 'detect':'strong positive'}

        t = image.threshold

        def created_by(s):
            if s.status.tag == "active":
                return mapping.get(s.created_by.tag)

            if s.created_by.tag == "detect" and s.status.tag == "deleted":
                d = s.created_by.contents
                if d.confidence >= t:
                    return "false positive"

        created = filter_none([created_by(s) for s in image.ann_summaries])
        d = count_dict(created)

        return d

    keys = ['false negative', 'weak positive', 'false positive']
    return image_histograms(dataset, make_histogram, keys=keys, n_splits=n_splits)
    

def action_histogram(dataset, n_splits=None):
    def make_histogram(image):
        summary = image_summary(image)
        return count_dict(pluck('action', summary.actions))
        
    keys = ['transform', 'confirm', 'add', 'delete', 'submit']
    return image_histograms(dataset, make_histogram, keys=keys, n_splits=n_splits)

     
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

    ax.set_title("Running mAP vs anotation time")

    ax.set_xlabel("Time (m)")
    ax.set_ylabel("mAP @" + iou)

    ax.legend()
    plt.show()


def cumulative_instances(datasets):

    fig, ax = plt.subplots(figsize=(24, 12))

    for k, dataset in datasets.items():
        summaries = image_summaries(dataset.history)

        instances = torch.Tensor(pluck('instances', summaries)).cumsum(0)
        durations = torch.Tensor(pluck('duration', summaries)).cumsum(0)

        plt.plot((durations / 60).numpy(), instances.numpy(), label = k)

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)

    ax.set_title("Cumulative annotated instances vs. time")

    ax.set_xlabel("Time (m)")
    ax.set_ylabel("Count")

    ax.legend()
    plt.show()


# def summary_figures(datasets):


    
base_path = '/home/oliver/storage/export/'


datasets = struct(
   penguins = 'penguins.json',
   # branches = 'branches.json',
    
   # seals = 'seals.json',
   # scott_base = 'scott_base.json',
   apples = 'apples.json',
   # fisheye = 'victor.json',
)

other = struct(
    branches    = 'mum/branches.json',
    seals       = 'seals_shanelle.json',
    scallops_niwa = 'scallops_niwa.json',
    scallops    = 'mum/scallops.json', 
    buoys       = 'mum/buoys.json',
)

penguins_dad = struct(
    hallett = 'dad/penguins_hallett.json',
    cotter = 'dad/penguins_cotter.json',
    royds = 'dad/penguins_royds.json',
)


penguins = struct(
    hallett = 'oliver/penguins_hallett.json',
    cotter = 'oliver/penguins_cotter.json',
    royds = 'oliver/penguins_royds.json',
    
)

if __name__ == '__main__':
    loaded = load_all(datasets, base_path)
    pprint_struct(pluck_struct('summary', loaded))


    annotation_histogram(loaded.apples)


    # loaded = load_all(penguins_dad, base_path)
    # pprint_struct(pluck_struct('summary', loaded))


    # loaded = load_all(penguins, base_path)
    # pprint_struct(pluck_struct('summary', loaded))


    # loaded = load_all(other, base_path)
    # pprint_struct(pluck_struct('summary', loaded))

    # actions_time(loaded)
    # plot_actions(loaded.apples)

    # oliver = load_all(penguins_oliver, base_path)
    # dad = load_all(penguins_dad, base_path)

    # cumulative_instances(loaded)
    # running_mAPs(loaded, iou=0.75)


    # load_all(penguins_oliver, base_path)
    # load_all(penguins_dad, base_path)