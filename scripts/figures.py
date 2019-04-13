import json
from dataset import annotate
from os import path

import argparse

from tools import struct, to_structs, filter_none, drop_while, concat_lists, map_dict, pluck
from detection import evaluate

from evaluate import compute_AP

import dateutil.parser as date

from matplotlib import rc
import matplotlib.pyplot as plt

import torch


import numpy as np
import matplotlib.pyplot as plt


def plot_stacks(x, stacks, keys, width):
    
    total = torch.Tensor(len(stacks)).zero_()
    bars = []
    for k in keys:
        values = pluck(k, stacks, 0)
        p = plt.bar(x, values, width, bottom=total.tolist())
        bars.append(p[0])
        total = total + torch.Tensor(values)
    plt.legend(bars, keys)


def plot_line_stacks(x, stacks, keys):
    
    total = torch.Tensor(len(stacks)).zero_()
    values = [pluck(k, stacks, 0) for k in keys]

    plt.stackplot(x, *values, labels=keys)
    plt.legend(loc='upper left')


def plot_cumulative_line_stacks(x, stacks, keys):
    
    total = torch.Tensor(len(stacks)).zero_()
    values = [np.cumsum(pluck(k, stacks, 0)) for k in keys]

    print(x)

    plt.stackplot(x, *values, labels=keys)
    plt.legend(loc='upper left')


def quartile_dict(k, data):
    l, lq, m, uq, u = data
    return {'med': m, 'q1': lq, 'q3': uq, 'whislo': l, 'whishi': u}

def make_legend(ax, colours, keys):

    legend = [Line2D([0], [0], color=colours, label=k) 
         for k in keys]
    ax.legend(handles=legend, loc='upper left')


def box_plot(quartiles, keys, colours):
    fig, ax = plt.subplots(figsize=(24, 12))

    for i, k in enumerate(keys):
        boxprops = dict(color=colours[k])

        d = quartile_dict(k, quartiles[k])
        plot = ax.bxp([d], positions=[i], widths=[0.5], vert=False, showfliers=False, boxprops=boxprops)

    ax.set_ylim(ymin=-0.5, ymax=len(keys)-0.5)

    plt.yticks(range(len(keys)), keys)
    return fig, ax