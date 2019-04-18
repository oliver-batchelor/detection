from scripts.history import history_summary, extract_histories, \
     image_summaries, running_mAP, image_summary, correction_types, action_types

from scripts.datasets import load_dataset, annotation_summary
from scripts.figures import *

from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection

import torch
import math

from tools import struct, to_structs, filter_none, drop_while, concat_lists, \
        map_dict, pprint_struct, pluck_struct, count_dict, sum_list, Struct, sum_dicts

from scripts.make_figures import *



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

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)

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
        plt.scatter(instances, duration, c=progress, s=actions*5, label = k, cmap='plasma')


    bar=plt.colorbar()
    bar.set_label('prior annotation time(min)')


    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)

    plt.xlabel("number of instances")
    plt.ylabel("annotation time (s)")

    plt.title('number of instances vs annotation time')

    return fig, ax





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

     
if __name__ == '__main__':
    figure_path = "/home/oliver/sync/figures/scatters"


    loaded = load_all(datasets, base_path)
    pprint_struct(pluck_struct('summary', loaded))


    #confidence_iou_scatter(loaded)
    plt.show()


