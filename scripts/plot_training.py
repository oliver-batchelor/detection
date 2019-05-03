from os import path
from tools import struct, to_structs, concat_lists, to_dicts, pluck, pprint_struct, transpose_structs, Struct, append_dict, transpose_dicts

from scripts.load_figures import *

import math
import json

import matplotlib.pyplot as plt
from collections import OrderedDict

import numpy as np
import torch

import pandas as pd
import xarray as xr
from scipy.signal import gaussian

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


def get_entries(log):
    entries = {}

    for i, entry in sortdict(log.steps):
        for k, v in entry.items():
            e = entries.get(k, []) 
            e.append((i, v))
            entries[k] = e

    return entries

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

def unzip(entries):
    return zip(*entries)


def best_epoch(key):
    def f(log):
        return max(get_entry(log, "validate"), key=lambda entry: entry[1][key])
    return f  

def plot_training_scatters(logs):
    fig, ax = make_chart()

    for k, log in logs.items():      
        train = get_entry(log, "train")

        validate = get_entry(log, "validate")
        split = get_entry(log, "validate_split")

        if len(split):
            _, AP_split = extract_key(split, 'AP')
            _, AP = extract_key(validate, 'AP')

            plt.scatter(AP_split, AP, label=k, color=dataset_colors[k], marker='.')

    plt.xlabel("tiling, average precision ($AP_{COCO}$)")
    plt.ylabel("full, average precision ($AP_{COCO}$)")

    plt.xlim(xmin=30)
    plt.ylim(ymin=30)

    plt.title("comparison of evaluation methods tiling vs. using the full image")

    plt.grid(True)
    unique_legend()

    return fig, ax




def plot_training_lines(logs):
    fig, ax = make_chart()

    for k, log in logs.items():      
        train = get_entry(log, "train")

        validate = get_entry(log, "validate")
        split = get_entry(log, "validate_split")

        if len(split):
            epoch, AP_split = extract_key(split, 'AP')
            _, AP = extract_key(validate, 'AP')

            plt.plot(epoch, AP, label=k, color=dataset_colors[k], linestyle='-')
            plt.plot(epoch, AP_split, label=k, color=dataset_colors[k], linestyle='--')

    plt.xlabel("training epoch")
    plt.ylabel("average precision ($AP_{COCO}$)")

    plt.title("comparison of evaluation methods tiling vs. using the full image")

    unique_legend()

    return fig, ax



log_files = struct(
    penguins = 'penguins',
    branches = 'branches',
    # seals1 = 'seals',
    # seals2 = 'seals_shanelle',
    scott_base = 'scott_base',
    # apples1 = 'apples',
    # apples2 = 'apples_lincoln',
    # scallops = 'scallops',  
    # fisheye = 'victor',
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


def unique_legend(**kwargs):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), **kwargs)    


def read_run(dataset, incremental, method, cycles, run):
    logfile = path.join(log_path, 'lr', run, incremental,  method, str(cycles), dataset, 'log.json')
    log = read_log(logfile)

    epoch, AP = extract_key(get_entry(log, "validate"), 'AP')
    _, loss = extract_key(get_entry(log, "train/loss"), 'total')
    examples = np.array(epoch) * cycles

    return struct(examples=examples, AP=np.array(AP), loss=np.array(loss))

    # if incremental == "full":
    #     scatters[dataset] += list(zip(AP, loss))

def plot_lr(figure_path):

    datasets = ["apples", "branches", "scallops"]
    methods = ['cosine', 'step', 'log']
    steps = [1024, 4096]

    colors = {}
    tab10 = plt.get_cmap("tab10")

    scatters = {dataset:[] for dataset in datasets}

    runs = ['0', '1', '2', '3', '4', '5']

    for dataset in datasets:
        fig, ax = make_chart()
        ax2 = ax.twinx()  

        aps = { (method, step) : [] for method in methods for step in steps}
        losses = { (method, step) : [] for method in methods for step in steps }        

        for incremental in ["incremental", "full"]:
            for method in methods:
                cycles_types = [1024] if method is 'step' else [1024, 4096]
                for cycles in cycles_types:
                    
                    results = [read_run(dataset, incremental, method, cycles, run) for run in runs]
                    if incremental == "full":
                        for r in results:
                            scatters[dataset] += list(zip(list(r.AP), list(r.loss)))

                            aps[(method, cycles)] += list(r.AP[len(r.AP)//2:])
                            losses[(method, cycles)] += list(r.loss[len(r.loss)//2:])

                    examples = results[0].examples
                    AP = sum([r.AP for r in results]) / len(results)
                    loss = sum([r.loss for r in results]) / len(results)

                    label = method if method is 'step' else method + "-" + str(cycles)
                    color = colors[label] if label in colors else tab10(len(colors))
                    colors[label] = color

                    style = '--' if incremental == "incremental" else '-'

                    ax.plot(examples, AP, label=label, linestyle=style, color=color)
                    ax2.plot(examples, loss, label=label, linestyle=style, color=color)

        print(dataset)
        for method in methods:
            cycles_types = [1024] if method is 'step' else [1024, 4096]

            for step in cycles_types:
                l = np.array(losses[(method, step)])
                ap = np.array(aps[(method, step)])

                print("loss", method, step, l.mean())
                print("AP", method, step, ap.mean(), ap.std())
                    
        plt.title("effect of learning rate scheduling on training " + dataset)
        ax.set_xlabel("training examples")
        ax.set_ylabel("average precision ($AP_{COCO}$), mean of " + str(len(runs)) + " runs")
        ax2.set_ylabel("training loss, mean of " + str(len(runs)) + " runs")

        plt.xlim(xmin=0)
        plt.ylim(ymin=0)

        unique_legend()

        fig.savefig(path.join(figure_path, "lr_schedule", dataset + ".pdf"), bbox_inches='tight')



    fig, ax = make_chart()
    for dataset, points in scatters.items():
        ap, loss = zip(*points)
        plt.scatter(np.array(loss) / min(loss), ap, label=dataset, marker='x')

    plt.xlim(xmin=1, xmax=5)
    plt.ylim(ymin=0)

    plt.xlabel("loss (percent factor of minimum)")
    plt.ylabel("average precision ($AP_{COCO}$)")

    plt.legend()

    fig.savefig(path.join(figure_path, "lr_schedule", "scatter_loss_ap.pdf"), bbox_inches='tight')


subsets_voc = struct(
    subset1=["cat",  "cow",  "dog",    "sheep"],
    subset2=["bicycle", "bus",  "car", "motorbike"],  
)

subsets_coco = struct(
    subset1=["cow",  "sheep", "cat",  "dog"],
    subset2=["zebra", "giraffe",  "elephant", "bear"],  
    subset3=["sandwich", "pizza",  "donut", "cake"],  
    subset4=["cup", "fork",  "knife", "spoon"],      
)


def plot_multiclass(figure_path, directory, subsets):
    fig, ax = make_chart()
                 
    tab10 = plt.get_cmap("tab20")
    all_classes = sum(subsets.values(), [])
    colors = {c : tab10(i) for i, c in enumerate(all_classes)}

    for subset, classes in subsets.items():

        combined_name = ','.join(classes)
        logfile = path.join(log_path, directory, combined_name, 'log.json')
        log = read_log(logfile)             

        for c in classes:
            class_file = path.join(log_path, directory, c, 'log.json')
            class_log = read_log(class_file)             

            epoch, ap_subset = extract_key(get_entry(log, "test/AP"), c)
            _, ap_class = extract_key(get_entry(class_log, "test"), 'AP')

            plt.plot(epoch, ap_subset, color=colors[c], linestyle='--', label=c)
            plt.plot(epoch, ap_class, color=colors[c], linestyle='-', label=c)


    plt.xlabel("epoch")
    plt.ylabel("class average precision ($AP_{COCO}$)")

    plt.title("training rate in single class and multi-class scenario")

    unique_legend(loc="upper right")
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
        fig, ax = make_chart()
        for s, scale in enumerate(scales):
            for crop in crops:

                logfile = path.join(log_path, 'scales', str(scale), str(crop), dataset, 'log.json')
                log = read_log(logfile)               

                epoch, AP = extract_key(get_entry(log, "validate"), 'AP')
                time = training_time(log)

                epoch = epoch[:40]
                AP = AP[:40]
                
                rows.append(struct(dataset=dataset, scale=1/scale, crop=crop, 
                    AP=np.array(AP[8:]).mean(), time=time[-1] / len(time)))

                plt.plot(epoch, AP,  color=colors(s), linestyle=styles[crop], label= str(1/scale * 100) + ":" + str(crop) )

        plt.title("effect of image scale and crop size on training - " + dataset)
        plt.xlabel("training epoch")
        plt.ylabel("average precision ($AP_{COCO}$)")

        plt.xlim(xmin=0)
        plt.ylim(ymin=0)

        plt.legend()

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
    fig, ax = make_chart()

    times = np.linspace(0, 3.999, num=800)

    plt.plot(times, list(map(log_anneal, times)),  label="log annealing")
    plt.plot(times, list(map(cosine_anneal, times)),  label="cosine annealing (SGDR)")
    plt.plot(times, list(map(step_func, times)),  label="step function")

    plt.xlabel("training time (epochs)")
    plt.ylabel("learning rate")

    plt.title("learning rate schedules")

    plt.legend()
    return fig, ax


tab10 = plt.get_cmap("tab10")
pr_colors = {k : tab10(i) for i, k in enumerate (["precision", "false positives", "false negatives", "true positives", "confidence"]) }


def plot_pr_curves(ax, prs):

    ax2 = ax.twinx()

    for i, pr in prs.items():
        recall = pr.recall
        print(i)

        suffix = lambda s: "$" + s + "_{" + str(i) + "}$"
        style="--" if i == 50 else "-"

        fpr = np.array(pr.false_positives)[1:] - np.array(pr.false_positives)[:-1]
        print(fpr)

        l1 = ax.plot(recall, pr.precision, label="precision", color=pr_colors["precision"], linestyle=style)
        l2 = ax2.plot(recall[1:], fpr, label="false positive ratio", color=pr_colors["false positives"], linestyle=style)
        # l3 = ax2.plot(recall, pr.true_positives, label="true positives", color=pr_colors["true positives"])
        # l4 = ax2.plot(recall, pr.false_negatives, label="false negatives", color=pr_colors["false negatives"])
        l5 = ax.plot(recall, pr.confidence, label="confidence", color=pr_colors["confidence"], linestyle=style)

    ax.set_xlim(xmin=0, xmax=1.0)
    ax.set_ylim(ymin=0, ymax=1.0)    
    ax2.set_ylim(ymin=0, ymax=10)    

    lines = sum([l1, l2,  l5], [])

    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="center left")
    


def plot_best_pr(log):

    APs = np.array(list(map(lambda entry: entry[1].AP, get_entry(log, "validate"))))

    _, pr50s = unzip(get_entry(log, "validate/pr50/total"))
    _, pr75s = unzip(get_entry(log, "validate/pr75/total"))


    i = np.argmax(APs)
    pr = {50 : pr50s[i], 75: pr75s[i]}
    

    fig, ax = make_chart()
    plot_pr_curves(ax, pr)
    return fig, ax



if __name__ == '__main__':

    figure_path = "/home/oliver/sync/figures/training"


    logs = read_logs(path.join(log_path, 'validate'), log_files)
    penguin_logs = read_logs('/home/oliver/storage/logs/penguins', penguins_a._merge(penguins_b))

    pprint_struct(penguin_logs._map(best_epoch('mAP50')))
    pprint_struct(logs._map(best_epoch('AP')))

    # fig, ax = plot_best_pr(logs.seals1)
    # fig, ax = plot_best_pr(logs.scott_base)

    # plt.show()


    # fig, ax = plot_training_lines(logs)
    # fig.savefig(path.join(figure_path, "splits.pdf"), bbox_inches='tight')

    # fig, ax = plot_training_scatters(logs)
    # fig.savefig(path.join(figure_path, "splits_scatters.pdf"), bbox_inches='tight')

    # plot_scales(figure_path)
    # plot_lr(figure_path)

    # fig, ax = plot_schedules()
    # fig.savefig(path.join(figure_path, "lr_schedules.pdf"), bbox_inches='tight')


    # fig, ax = plot_multiclass(figure_path, 'multiclass', subsets_voc)
    # fig.savefig(path.join(figure_path, "multiclass.pdf"), bbox_inches='tight')


    fig, ax = plot_multiclass(figure_path, 'multiclass_coco', subsets_coco)
    fig.savefig(path.join(figure_path, "multiclass_coco.pdf"), bbox_inches='tight')
