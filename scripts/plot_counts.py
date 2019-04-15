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

import csv

base_path = '/home/oliver/storage/export/'


def subset(text, image_counts):
    return [count for count in image_counts if text in count.image_file]

def plot_estimate(images, colour):
    estimates = transpose_structs(pluck('estimate', images))
    times = pluck('time', images)

    mask = torch.ByteTensor([1 if i.category != 'discard' else 0 for i in images])

    def f(xs):
        return window.masked_mean(torch.Tensor(xs), mask=mask, window=7, clamp=False).numpy()

    # middle = window.rolling_window(torch.Tensor(estimates.middle), window=5).mean(1).numpy()
    estimates = estimates._map(f)
    
    plt.plot(times, estimates.middle, colour)
    plt.fill_between(times, estimates.upper, estimates.lower, facecolor=colour, alpha=0.4)

def plot_points(images, colour, marker, key=lambda i: i.truth):
    truth = list(map(key, images))
    times = pluck('time', images)

    plt.plot(times, truth, marker, markeredgecolor=colour, markersize=8)


def pick(images, classes):
    return [i for i in images if i.category in classes]


def plot_subset(images, colour):

    plot_estimate(images, colour)

    plot_points(pick(images, ['train']), colour,   '+')
    plot_points(pick(images, ['validate']), colour, 'x')
    plot_points(pick(images, ['discard']), colour,  'rX', key=lambda i: i.estimate.middle)

    plot_points(pick(images, ['test']), colour,    'gP')


def plot_runs(*runs, loc='upper left'):
  
    def run_legend(run):
        return Line2D([0], [0], color=run.colour, label=run.label)

    legend = list(map(run_legend, runs)) + [
        Line2D([0], [0], marker='P', color='g', markeredgecolor='y', linestyle='None', label='test'),

        Line2D([0], [0], marker='+', color='y',  markeredgecolor='y', linestyle='None', label='train'),
        Line2D([0], [0], marker='x', color='y', markeredgecolor='y', linestyle='None', label='validate'),

        Line2D([0], [0], marker='X', color='r', markeredgecolor='y', linestyle='None', label='discard')
    ]

    fig, ax = plt.subplots(figsize=(24, 12))



    plt.xlabel("Date")
    plt.ylabel("Count")

    plt.gcf().autofmt_xdate()

    for run in runs:
        plot_subset(run.data, run.colour)

    ax.set_ylim(ymin=0)


    ax.legend(handles=legend, loc=loc)
    return fig




def load(filename):
    return load_dataset(path.join(base_path, filename))


figure_path = "/home/oliver/sync/figures/seals/"


datasets = struct(
    scott_base = 'scott_base.json',
    scott_base_100 = 'scott_base_100.json',
    seals      = 'seals.json',
    seals_102  = 'seals_102.json',

    seals_shanelle  = 'seals_shanelle.json',
)


def flatten_dict(dd, separator='_', prefix=''):
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }


def export_csv(file, fields, rows):

    with open(file, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row._to_dicts())

def export_counts(file, counts):

    fields = ['image_file', 'time', 'truth', 'category', 'lower', 'estimate', 'upper']

    def f(counts):
        return struct(
            image_file=counts.image_file, 
            time= counts.time.strftime("%Y-%m-%d %H:%M:%S"), 
            truth=None if counts.category=='new' else counts.truth, 
            category=counts.category,
            lower = counts.estimate.upper,
            estimate  = counts.estimate.middle,
            upper = counts.estimate.lower
        )

    export_csv(file, fields, list(map(f, counts)))

def plot_counts(loaded):
    for k in ['scott_base', 'scott_base_100']:
        scott_base = get_counts(loaded[k])

        cam_b  = subset("CamB", scott_base)
        cam_c  = subset("CamC", scott_base)

        fig = plot_runs(
            struct(data = cam_b, colour='g', label="camera b"),
            struct(data = cam_c, colour='y', label="camera c" ),
        )

        fig.savefig(path.join(figure_path, k + ".pdf"), bbox_inches='tight')
        export_counts(path.join(figure_path, k + "_cam_b.csv"), cam_b)
        export_counts(path.join(figure_path, k + "_cam_c.csv"), cam_c)


    for k in ['seals', 'seals_102', 'seals_shanelle']:
        seals_total = get_counts(loaded[k])
        seals_pairs = get_counts(loaded[k], class_id = 1)

        fig = plot_runs(
            struct(data = seals_total, colour='y', label="total"),
            struct(data = seals_pairs, colour='c', label="pairs"),

            loc='upper right'
        )

        fig.savefig(path.join(figure_path, k + ".pdf"), bbox_inches='tight')
        export_counts(path.join(figure_path, k + ".csv"), seals_total)
        export_counts(path.join(figure_path, k + "_pairs.csv"), seals_pairs)   


def show_errors(loaded):

    # print ("--------" + k + "--------")
    truth = {image.image_file:image.truth
        for image in get_counts(loaded['seals']) if image.category=='test'}

    truth2 = {image.image_file:image.truth
        for image in get_counts(loaded['seals_shanelle']) if image.category=='test'}

    estimate = {image.image_file:image.estimate.middle 
        for image in get_counts(loaded['seals']) if image.category=='test'}

    # [(k, truth[k] - estimate[k], truth[k] - truth2[k]) for k in truth.keys()]

    errors = struct (
        human_human = [abs(truth[k] - truth2[k]) for k in truth.keys()],
        human_estimate = [abs(truth[k] - estimate[k]) for k in truth.keys()],
        human_estimate2 = [abs(truth2[k] - estimate[k]) for k in truth.keys()]
    )

    print(errors._map(np.mean))


    

if __name__ == '__main__':
      
    loaded = datasets._map(load)


    show_errors(loaded)




    # plot_counts(path.join())