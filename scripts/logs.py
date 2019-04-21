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
        step[entry.tag] = entry.value

        tags[entry.tag] = True
        steps[entry.step] = step

    return struct (tags=tags.keys(), steps={i : Struct(entry) for i, entry in steps.items()})





def sortdict(d, **opts):
    # **opts so any currently supported sorted() options can be passed
    for k in sorted(d, **opts):
        yield k, d[k]

def get_entry(log, name):
    return {i:entry[name] for i, entry in log.steps.items() if name in entry}




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
    return zip(*sorted(transpose_dicts(entries)[key].items()))


def best_epoch(key):
    def f(log):
        return max(get_entry(log, "validate").items(), key=lambda entry: entry[1][key])
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


def plot_scales(figure_path):

    scale_logs = [(scale, crop, read_logs(path.join(log_path, 'scales', str(scale), str(crop)), log_files))
        for scale in [2,4,8,16]
            for crop in [512, 768, 1024]]    


def plot_lr(figure_path):
    lr_logs = [(method, cycle, read_logs(path.join(log_path, 'lr', method, str(cycle)), log_files))
        for method in ['cosine', 'step', 'log']
            for cycle in [1024, 2048, 4096]]    



if __name__ == '__main__':

    figure_path = "/home/oliver/sync/figures/training"


    # logs = read_logs(path.join(log_path, 'validate'), log_files)
    # penguin_logs = read_logs('/home/oliver/storage/logs/penguins', penguins_a._merge(penguins_b))

    # pprint_struct(penguin_logs._map(best_epoch('mAP50')))
    # pprint_struct(logs._map(best_epoch('AP')))

    # fig, ax = plot_training_lines(logs)
    # fig.savefig(path.join(figure_path, "splits.pdf"), bbox_inches='tight')

    # plot_scales(figure_path)
    plot_lr(figure_path)


    # for method in 
    #   for cycle in [1, 2, 4]]