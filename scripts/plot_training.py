from os import path
from tools import struct, to_structs, concat_lists, to_dicts, pluck, pprint_struct, transpose_structs, Struct, append_dict, transpose_dicts

from scripts.load_figures import dataset_colors

import math
import json

import matplotlib.pyplot as plt
from collections import OrderedDict

import numpy as np
import torch

import pandas as pd
import xarray as xr


def read_log(file):
    entries = [to_structs(json.loads(line)) for line in open(file, mode="r")]

    steps = {}
    tags = {}

    for entry in entries:
        step = steps.get(entry.step) or {}
        step[entry.tag] = struct(value=entry.value, time=entry.time)

        tags[entry.tag] = True
        steps[entry.step] = step

    return struct (tags=tags.keys(), steps={i : Struct(step) for i, step in steps.items()})





def sortdict(d, **opts):
    # **opts so any currently supported sorted() options can be passed
    for k in sorted(d, **opts):
        yield k, d[k]

def get_entry(log, name):
    return [(i, entry[name].value) for i, entry in sortdict(log.steps) if name in entry]


def get_entry_time(log, name):
    return [(entry[name].time, entry[name].value) for i, entry in sortdict(log.steps) if name in entry]



def get_prefixed(log, prefix):
    keys = [tag[len(prefix) + 1:] for tag in log.tags if tag.startswith(prefix)]
    entries = {k:get_entry(log, prefix + "/" + k) for k in keys}
    return transpose_dicts(entries)


def get_prs(log, category='validate'):
    return get_keys(log, category + "/pr")
    



def read_logs(base_path, log_files):

    def load(run_name):
        filename = path.join(base_path, run_name, "log.json")
        print(filename)

        if path.isfile(filename):
            return read_log(filename)

    return log_files._map(load)._filter_none()


def extract_key(entries, key):
    i, values = zip(*entries)
    return i, [value[key] for value in values]



def best_epoch(key):
    def f(log):
        return max(get_entry(log, "validate"), key=lambda entry: entry[1][key])
    return f  


def plot_training_lines(logs):
    fig, ax = plt.subplots(figsize=(24, 12))

    for k, log in logs.items():      
        train = get_entry(log, "train")

        validate = get_entry(log, "validate")
        split = get_entry(log, "validate_split")

        if len(split):
            epoch, AP_split = extract_key(split, 'AP')
            _, AP = extract_key(validate, 'AP')

            plt.plot(epoch, AP, label=k + " full", color=dataset_colors[k], linestyle='-')
            plt.plot(epoch, AP_split, label=k + " split", color=dataset_colors[k], linestyle='--')

    plt.xlabel("training epoch")
    plt.ylabel("average precision (AP)")

    plt.title("comparison of evaluation by splitting and using the full image")

    plt.legend()

    return fig, ax



log_files = struct(
    penguins = 'penguins',
    branches = 'branches',
    seals = 'seals',
    scott_base = 'scott_base',
    apples1 = 'apples',
    apples2 = 'apples_lincoln',
    scallops = 'scallops',  
    fisheye = 'victor',
    buoys       = 'buoys',
    aerial_penguins = 'aerial_penguins'
)
 
penguins_a = struct(
    hallett_a = 'oliver_hallett',
    cotter_a = 'oliver_cotter',
    royds_a = 'oliver_royds',
)

penguins_b = struct(
    hallett_b = 'dad_hallett',
    cotter_b = 'dad_cotter',
    royds_b = 'dad_royds',
)

log_path = '/home/oliver/storage/logs/'


def unique_legend():
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())    

def plot_lr(figure_path):

    datasets = ["apples", "branches", "scallops"]

    colors = {}
    tab10 = plt.get_cmap("tab10")

    scatters = {dataset:[] for dataset in datasets}

    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(24, 12))
        ax2 = ax.twinx()  

        for incremental in ["incremental", "full"]:
            for method in ['cosine', 'step', 'log']:
                cycles_types = [1024] if method is 'step' else [1024, 4096]
                for cycles in cycles_types:
                    logfile = path.join(log_path, 'lr', incremental,  method, str(cycles), dataset, 'log.json')
                    log = read_log(logfile)

                    epoch, AP = extract_key(get_entry(log, "validate"), 'AP')
                    epoch, loss = extract_key(get_entry(log, "train/loss"), 'total')
                    examples = np.array(epoch) * cycles

                    label = method if method is 'step' else method + "-" + str(cycles)
                    color = colors[label] if label in colors else tab10(len(colors))
                    colors[label] = color

                    if incremental == "full":
                        scatters[dataset] += list(zip(AP, loss))

                    style = '--' if incremental == "incremental" else '-'

                    ax.plot(examples, AP, label=label, linestyle=style, color=color)
                    ax2.plot(examples, loss, label=label, linestyle=style, color=color)

                    
        plt.title("effect of learning rate scheduling on training " + dataset)
        ax.set_xlabel("training examples")
        ax.set_ylabel("average precision ($AP_{COCO}$)")
        ax2.set_ylabel("training loss")

        plt.xlim(xmin=0)
        plt.ylim(ymin=0)

        unique_legend()

        fig.savefig(path.join(figure_path, incremental + "_" + dataset + ".pdf"), bbox_inches='tight')


    fig, ax = plt.subplots(figsize=(24, 12))
    for dataset, points in scatters.items():
        ap, loss = zip(*points)
        plt.scatter(np.array(loss) / min(loss), ap, label=dataset, marker='x')

    plt.xlim(xmin=1, xmax=5)
    plt.ylim(ymin=0)

    plt.xlabel("loss (percent factor of minimum)")
    plt.ylabel("average precision ($AP_{COCO}$)")

    plt.legend()

    fig.savefig(path.join(figure_path, "scatter_loss_ap.pdf"), bbox_inches='tight')


def plot_multiclass(figure_path):
    fig, ax = plt.subplots(figsize=(24, 12))

    subsets = struct(
        subset1=["cat",  "cow",  "dog",    "sheep"],
        subset2=["bicycle", "bus",  "car", "motorbike"],  
    )
                 
    tab10 = plt.get_cmap("tab10")
    all_classes = sum(subsets.values(), [])
    colors = {c : tab10(i) for i, c in enumerate(all_classes)}

    for subset, classes in subsets.items():
        logfile = path.join(log_path, 'multiclass', subset, 'log.json')
        log = read_log(logfile)             

        for c in classes:
            class_file = path.join(log_path, 'multiclass', c, 'log.json')
            class_log = read_log(class_file)             

            epoch, ap_subset = extract_key(get_entry(log, "test/AP"), c)
            _, ap_class = extract_key(get_entry(class_log, "test"), 'AP')

            plt.plot(epoch, ap_subset, color=colors[c], linestyle='--', label=c)
            plt.plot(epoch, ap_class, color=colors[c], linestyle='-', label=c)


    plt.xlabel("epoch")
    plt.ylabel("class average precision ($AP_{COCO}$)")

    plt.title("training rate in single class and multi-class scenario")

    unique_legend()
    return fig, ax

def training_time(log):
    start_times, _ = zip(*get_entry_time(log, "dataset"))
    train_times, _ = zip(*get_entry_time(log, "train/loss"))

    durations = np.array(train_times) - np.array(start_times)
    return np.cumsum(durations) / 60

def plot_scales(figure_path):
    datasets = ["apples", "penguins", "scallops", "seals"]

    scales = [1,2,4,8]
    crops = [512, 768, 1024]

    colors = plt.get_cmap("tab10")
    styles = {512: ':', 1024:'-', 768:'--'}

    rows = []
    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(24, 12))
        for s, scale in enumerate(scales):
            for crop in crops:

                logfile = path.join(log_path, 'scales', str(scale), str(crop), dataset, 'log.json')
                log = read_log(logfile)               

                _, AP = extract_key(get_entry_time(log, "validate"), 'AP')
                time = training_time(log)
                
                rows.append(struct(dataset=dataset, scale=1/scale, crop=crop, 
                    AP=np.array(AP[8:]).mean(), time=time[-1] / len(time)))

                plt.plot(time, AP,  color=colors(s), linestyle=styles[crop], label=dataset + " " )

        plt.title("effect of image scale and crop size on training - " + dataset)
        plt.xlabel("training time (minutes)")
        plt.ylabel("average precision ($AP_{COCO}$)")

        plt.xlim(xmin=0)
        plt.ylim(ymin=0)

        fig.savefig(path.join(figure_path, "crops_scales", dataset + ".pdf"), bbox_inches='tight')


    df = pd.DataFrame(rows)
    aps = []
    times = []

    for dataset in datasets:
        d = df.loc[df['dataset'] == dataset]
        
        ap = d.pivot(columns='scale', index='crop', values='AP')
        time = d.pivot(columns='scale', index='crop', values='time')
        
        
        aps.append(ap/ap.max().max())
        times.append(time.max().max()/time)
        
    print(sum(aps) / len(aps))
    print(sum(times) / len(times))


            


def log_anneal(e):
    t = math.fmod(e, 1)
    begin, end = 0.1, 0.01
    return math.exp(math.log(begin) * (1 - t) + math.log(end) * t)

def cosine_anneal(e):
    t = math.fmod(e, 1)
    begin, end = 0.1, 0.01
    return end + 0.5 * (begin - end) * (1 + math.cos(t * math.pi))

def step_func(e):
    return 0.01 if e > 2 else 0.1

def plot_schedules():
    fig, ax = plt.subplots(figsize=(24, 12))

    times = np.linspace(0, 3.999, num=800)

    plt.plot(times, list(map(log_anneal, times)),  label="log annealing")
    plt.plot(times, list(map(cosine_anneal, times)),  label="cosine annealing (SGDR)")
    plt.plot(times, list(map(step_func, times)),  label="step function")

    plt.xlabel("training time (epochs)")
    plt.ylabel("learning rate")

    plt.title("learning rate schedules")

    plt.legend()
    return fig, ax




        



if __name__ == '__main__':

    figure_path = "/home/oliver/sync/figures/training"


    # logs = read_logs(path.join(log_path, 'validate'), log_files)
    # penguin_logs = read_logs('/home/oliver/storage/logs/penguins', penguins_a._merge(penguins_b))

    # pprint_struct(penguin_logs._map(best_epoch('mAP50')))
    # pprint_struct(logs._map(best_epoch('AP')))

    # fig, ax = plot_training_lines(logs)
    # fig.savefig(path.join(figure_path, "splits.pdf"), bbox_inches='tight')

    # plot_scales(figure_path)
    # plot_lr(figure_path)

    # fig, ax = plot_schedules()
    # fig.savefig(path.join(figure_path, "lr_schedules.pdf"), bbox_inches='tight')


    fig, ax = plot_multiclass(figure_path)
    fig.savefig(path.join(figure_path, "multiclass.pdf"), bbox_inches='tight')

    plt.show()