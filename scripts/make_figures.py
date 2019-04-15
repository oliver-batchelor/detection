from scripts.history import history_summary, extract_histories, \
     image_summaries, running_mAP, image_summary

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
    # pprint_struct(pluck_struct('summary', loaded))

    
# def dataset_size(datasets):
#     fig, ax = plt.subplots(figsize=(24, 12))

#     for k in sorted(datasets.keys()):
#         dataset = datasets[k]

#         instances = np.array(pluck('instances', summaries))
#         instances = np.array(pluck('n_images', summaries))
        

#         summary = 


def actions_time_scatter(datasets):

    fig, ax = plt.subplots(figsize=(24, 12))

    for k in sorted(datasets.keys()):
        dataset = datasets[k]

        summaries = image_summaries(dataset.history)

        instances = np.array(pluck('instances', summaries))
        duration = np.array(pluck('duration', summaries))
        actions = np.array(pluck('n_actions', summaries))

        plt.scatter(actions, duration, s=instances * 5, alpha=0.5, label = k)


    def update(handle, orig):
        handle.update_from(orig)
        handle.set_sizes([64])

    plt.legend(handler_map={PathCollection : HandlerPathCollection(update_func=update)})

    # ax.set_ylim(ymin=0, ymax=150)
    # ax.set_xlim(xmin=0, xmax=50)

    plt.xlabel("number of actions")
    plt.ylabel("annotation duration")

    return fig, ax


def instances_duration_scatter(datasets, keys):

    fig, ax = plt.subplots(figsize=(24, 12))

    for k in keys:
        dataset = datasets[k]

        summaries = image_summaries(dataset.history)

        instances = np.array(pluck('instances', summaries))
        duration = np.array(pluck('duration', summaries))
        actions = np.array(pluck('n_actions', summaries))

        progress = np.cumsum(duration) / 60
        # progress = progress / np.amax(progress)

        plt.scatter(instances, duration, c=progress, s=actions*5, label = k, cmap='plasma')


    bar=plt.colorbar()
    bar.set_label('prior annotation time(min)')


    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)

    plt.xlabel("number of instances")
    plt.ylabel("annotation time (s)")

    plt.title('number of instances vs annotation time')

    return fig, ax





def sum_splits(xs, n_splits=None):
    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    return list(map(sum_dicts, split(xs, n_splits)))

def image_histograms(dataset, get_values, keys, n_splits=None):
    actions = [get_values(image) for image in dataset.history]        

    if n_splits is not None:
        actions = sum_splits(actions, n_splits=n_splits)

    n = len(actions)
    
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_stacks(np.array(range(n)) + 0.5, actions, keys, width= 0.5)
    plt.show()    



def cumulative_lines(dataset, get_values, keys):

    actions = [get_values(image) for image in dataset.history]        
    durations = torch.Tensor(pluck('duration', dataset.history)).cumsum(0)

    fig, ax = plt.subplots(figsize=(24, 12))
    plot_cumulative_line_stacks(durations, actions, keys)
    plt.show()    


def thresholds(image):
    changes = [action.value for action in image.actions if action.action=='threshold']
    return [image.threshold] + changes




def annotation_lines(dataset):
    return cumulative_lines(dataset, annotation_categories, keys=annotation_types)

def annotation_histogram(dataset, n_splits=None):
    return image_histograms(dataset, annotation_categories, keys=annotation_types, n_splits=n_splits)


def area(box):
    lx, ly = box.lower
    ux, uy = box.upper

    return (max(0, ux - lx)) * (max(0, uy - ly))


def iou_box(box1, box2):
    overlap = struct(lower = np.maximum(box1.lower, box2.lower),  
    upper = np.minimum(box1.upper, box2.upper))

    i = area(overlap)
    u = (area(box1) + area(box2) - i)
    
    return i / u

def bounds_circle(c):
    r = np.array([c.radius, c.radius])
    return struct(lower = np.array(c.centre) - r, upper = np.array(c.centre) + r)

def iou_circle(c1, c2):
    return iou_box(bounds_circle(c1), bounds_circle(c2))

def iou_shape(shape1, shape2):
    assert shape1.tag == shape2.tag, shape1.tag + ", " + shape2.tag

    if shape1.tag == "box":
        return iou_box(shape1.contents, shape2.contents)
    elif shape1.tag == "circle":
        return iou_circle(shape1.contents, shape2.contents)
    else:
        assert False, "unknown shape: " + shape1.tag



def confidence_iou_scatter(datasets):
    fig, ax = plt.subplots(figsize=(24, 12))

    def get_point(s):
        if s.status.tag == "active" and s.created_by.tag in ["detect", "confirm"]:
            detection = s.created_by.contents
            ann = s.status.contents

            return (detection.confidence, iou_shape(detection.shape, ann.shape))

    for k, dataset in datasets.items():
        points = filter_none([get_point(s) for image in dataset.history 
            for s in image.ann_summaries])

        conf, iou = zip(*points)

        plt.scatter(conf, iou, alpha=0.4, label = k)

    plt.legend()
    plt.show()

action_types = ['transform', 'confirm', 'add', 'delete', 'submit']

def action_histogram(dataset, n_splits=None):
    def make_histogram(image):
        summary = image_summary(image)
        return count_dict(pluck('action', summary.actions))
        
    return image_histograms(dataset, make_histogram, keys=action_types, n_splits=n_splits)

     
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
colour_map = {k : tab10(i) for i, k in enumerate (datasets.keys()) }

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