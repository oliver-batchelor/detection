from scripts.datasets import load_dataset, get_counts
from os import path

from tools import *
import tools.window as window


import datetime
import random

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt

import numpy as np
import torch


base_path = '/home/oliver/storage/export/'


def subset(text, image_counts):
    return [count for count in image_counts if text in count.image_file]

def plot_estimate(images, colour):
    estimates = transpose_structs(pluck('estimate', images))
    times = pluck('time', images)

    mask = torch.ByteTensor([1 if i.category != 'discard' else 0 for i in images])

    def f(xs):
        return window.masked_mean(torch.Tensor(xs), mask=mask, window=5, clamp=False).numpy()

    # middle = window.rolling_window(torch.Tensor(estimates.middle), window=5).mean(1).numpy()
    estimates = estimates._map(f)
    
    plt.plot(times, estimates.middle, colour)
    plt.fill_between(times, estimates.upper, estimates.lower, facecolor=colour, alpha=0.4)

def plot_points(images, marker, key=lambda i: i.truth):
    truth = list(map(key, images))
    times = pluck('time', images)

    plt.plot(times, truth, marker, markersize=8)


def pick(images, classes):
    return [i for i in images if i.category in classes]


def plot_subset(images, colour):

    plot_estimate(images, colour)

    plot_points(pick(images, ['test']), colour + 'D')
    plot_points(pick(images, ['train']), colour + '+')
    plot_points(pick(images, ['validate']), colour + 'o')
    plot_points(pick(images, ['discard']), colour + 'X')


def plot_runs(*runs):
  
    def run_legend(run):
        return Line2D([0], [0], color=run.colour, label=run.label)

    legend = list(map(run_legend, runs)) + [
        Line2D([0], [0], marker='+', color='black', linestyle='None', label='train'),
        Line2D([0], [0], marker='X', color='black', linestyle='None', label='discard'),

        Line2D([0], [0], marker='o', color='black', linestyle='None', label='validate'),
        Line2D([0], [0], marker='D', color='black', linestyle='None', label='test'),
    ]

    fig, ax = plt.subplots()

    plt.xlabel("Date")
    plt.ylabel("Count")

    plt.gcf().autofmt_xdate()

    for run in runs:
        plot_subset(run.data, run.colour)
    

    ax.legend(handles=legend, loc='upper left')
    plt.show()



def load(filename):
    dataset = load_dataset(path.join(base_path, filename))
    image_counts = get_counts(dataset)




datasets = struct(
    scott_base = 'scott_base.json',
)

if __name__ == '__main__':
      

    loaded = datasets._map(load)

    cam_c  = subset("CamB", image_counts)
    cam_b  = subset("CamC", image_counts)

    plot_runs(
        struct(data = cam_b, color='g', label="camera b"),
        struct(data = cam_c, color='y', label="camera c" )
    )



    # plot_counts(path.join())