from os import path
from tools import struct, to_structs, concat_lists, to_dicts, pluck, pprint_struct, transpose_structs, Struct, append_dict, transpose_dicts

from scripts.load_figures import dataset_colors

import math
import json

import matplotlib.pyplot as plt

import numpy as np
import torch


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


def plot_lr(figure_path):
    fig, ax = plt.subplots(figsize=(24, 12))

    datasets = ["apples", "branches"]
    for dataset in datasets:
        for method in ['cosine', 'step', 'log']:
            cycles_types = [1024,  4096] if method is not 'step' else [1024]
            for cycles in cycles_types:
                logfile = path.join(log_path, 'lr', method, str(cycles), dataset, 'log.json')
                log = read_log(logfile)

                epoch, loss = extract_key(get_entry(log, "train/loss"), 'total')
                examples = np.array(epoch) * cycles

                label = method if method is 'step' else method + "-" + str(cycles)
                plt.plot(np.array(epoch) * cycles, loss, label=label)
                
        plt.title("effect of learning rate scheduling on training - " + dataset)
        plt.xlabel("training examples")
        plt.ylabel("training loss")

        fig.savefig(path.join(figure_path, "lr_schedule", dataset + ".pdf"), bbox_inches='tight')


def plot_scales(figure_path):
    datasets = ["apples", "penguins", "scallops", "seals"]

    scales = [2,4,8]
    crops = [512, 768, 1024]

    colors = plt.get_cmap("rainbow")
    styles = {512: ':', 1024:'-', 768:'--'}

    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(24, 12))
        for s, scale in enumerate(scales):
            for crop in crops:

                logfile = path.join(log_path, 'scales', str(scale), str(crop), dataset, 'log.json')
                log = read_log(logfile)                

                time, AP = extract_key(get_entry_time(log, "validate"), 'AP')
                time = (np.array(time) - time[0]) / 60             

                plt.plot(time, AP,  color=colors(s / len(scales)), linestyle=styles[crop])

                plt.title("effect of image scale and crop size on training - " + dataset)
                plt.xlabel("training time (minutes)")
                plt.ylabel("average precision (AP)")

                fig.savefig(path.join(figure_path, "crops_scales", dataset + ".pdf"), bbox_inches='tight')


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


    logs = read_logs(path.join(log_path, 'validate'), log_files)
    penguin_logs = read_logs('/home/oliver/storage/logs/penguins', penguins_a._merge(penguins_b))

    pprint_struct(penguin_logs._map(best_epoch('mAP50')))
    pprint_struct(logs._map(best_epoch('AP')))

    fig, ax = plot_training_lines(logs)
    fig.savefig(path.join(figure_path, "splits.pdf"), bbox_inches='tight')

    plot_scales(figure_path)
    # plot_lr(figure_path)

    fig, ax = plot_schedules()
    fig.savefig(path.join(figure_path, "lr_schedules.pdf"), bbox_inches='tight')

