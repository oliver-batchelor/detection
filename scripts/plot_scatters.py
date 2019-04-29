
from scripts.datasets import load_dataset, annotation_summary
from scripts.figures import *
from scripts.history import *


from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection

import torch
import math

from tools import struct, to_structs, filter_none, drop_while, concat_lists, \
        map_dict, pprint_struct, pluck_struct, count_dict, sum_list, Struct, sum_dicts

from scripts.load_figures import *
from scipy import stats



def actions_time_scatter(datasets, color_map):

    fig, ax = make_chart()

    for k in sorted(datasets.keys()):
        dataset = datasets[k]

        summaries = image_summaries(dataset.history)

        instances = np.array(pluck('instances', summaries))
        duration = np.array(pluck('duration', summaries))
        actions = np.array(pluck('n_actions', summaries))

        plt.scatter(actions, duration, s=instances * 5, alpha=0.5, label = k, color=color_map[k])


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

    fig, ax = make_chart()

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

def get_annotation_ious(dataset):
    
    def get_point(s):
        if s.status.tag == "active" and s.created_by.tag in ["detect", "confirm"]:
            detection = s.created_by.contents
            ann = s.status.contents

            return (detection.confidence, iou_shape(detection.shape, ann.shape), s.created_by.tag)

    return filter_none([get_point(s) for image in dataset.history 
            for s in image.ann_summaries])


def iou_lines(datasets, keys, color_map):
    fig, ax = make_chart()

    for k in keys:
        dataset = datasets[k]

        conf, iou, label = zip (*[point for point in get_annotation_ious(dataset) if point[1] < 1.0])

        density = stats.kde.gaussian_kde(iou)
    
        x = np.arange(0., 1.001, .002)
        plt.plot(x, density(x), label = k, color = color_map[k])

    plt.xlabel('iou')
    plt.ylabel('density')

    plt.title('prediction iou with respect to final annotation (modified annotations only)')

    plt.legend(loc = 'upper left')

    return fig, ax



def confidence_iou_scatter(datasets):
    fig, ax = make_chart()

    points = [p for dataset in datasets.values() 
        for p in get_annotation_ious(dataset)]

    conf, iou, label = zip(*points)

    xrange = np.arange(0.2, 1.001, 0.05)
    yrange = np.arange(0.4, 1.001, 0.05)
    hist, xbins, ybins, im = ax.hist2d(np.array(conf) - 0.01, np.array(iou) - 0.01, bins=(xrange, yrange), vmax=200)

    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            ax.text(xbins[j] + 0.025,ybins[i] + 0.025, int(hist[j,i]), 
                    color="w", ha="center", va="center")

    plt.xlabel('prediction confidence')
    plt.ylabel('iou with final annotation')

    plt.title('annotation iou vs prediction confidence')
   
    return fig, ax

     
if __name__ == '__main__':
    figure_path = "/home/oliver/sync/figures/scatters"

    loaded = load_all(datasets, base_path)
    pprint_struct(pluck_struct('summary', loaded))

    fig, ax = confidence_iou_scatter(loaded._subset('seals1', 'seals2', 'apples1', 'apples2', 'penguins', 'fisheye', 'branches', 'scallops'))
    fig.savefig(path.join(figure_path, "confidence_iou.pdf"), bbox_inches='tight')

    fig, ax = confidence_iou_scatter(loaded._subset('aerial_penguins', 'scott_base'))
    fig.savefig(path.join(figure_path, "confidence_iou_small.pdf"), bbox_inches='tight')    

    fig, ax = iou_lines(loaded, keys=sorted(loaded.keys()), color_map=dataset_colors)
    fig.savefig(path.join(figure_path, "iou_dataset.pdf"), bbox_inches='tight')
    
    fig, ax = actions_time_scatter(loaded._subset('apples1', 'apples2'), color_map=dataset_colors)
    fig.savefig(path.join(figure_path, "apples_scatter.pdf"), bbox_inches='tight')

    fig, ax = actions_time_scatter(loaded._subset('seals1', 'seals2'), color_map=dataset_colors)
    fig.savefig(path.join(figure_path, "seals_scatter.pdf"), bbox_inches='tight')