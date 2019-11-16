
from scripts.load_figures import make_chart, paired
from scripts.datasets import load_dataset, get_counts
from os import path

from tools import *
import tools.window as window

from datetime import datetime, timedelta
from dateutil.tz import tzutc


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


def sum_estimates(estimates):

    start = min (e.times[0] for e in estimates)
    end = max (e.times[-1] for e in estimates)

    eval_times = np.linspace(start.timestamp(), end.timestamp(), 36000)

    def interp(times, keys):
        times = [t.timestamp() for t in times]
        return np.interp(eval_times, times, keys)


    d = {k:sum([interp(e.times, e[k]) for e in estimates]) for k in ['middle', 'lower', 'upper']}

    times = [datetime.fromtimestamp(t, start.tzinfo) for t in eval_times]
    return Struct(d)._extend(times = times)


    # t = estimates[0].times[0]
    # s = t.timestamp()

    # print(t, datetime.fromtimestamp(s, t.tzinfo))


def fill_gaps(images, delta):
    min_gap = delta * 2
    r = []

    for i in range(len(images) - 1):
        r.append(images[i])

        t = images[i].time
        next = images[i + 1].time

        while (next - t > min_gap):
            t += delta
            r.append(struct(time = t, category='discard', estimate=images[i].estimate))

    r.append(images[-1])
    return r



def get_estimates(images, delta=None):
    if delta is not None:
        images = fill_gaps(images, delta)

    estimate_points = transpose_structs(pluck('estimate', images))
    times = pluck('time', images)

    mask = torch.ByteTensor([1 if i.category != 'discard' else 0 for i in images])
    def f(xs):
        return window.masked_mean(torch.Tensor(xs), mask=mask, window=7, clamp=False).numpy()

    estimate_points = estimate_points._map(f)
    return estimate_points._extend(times=times)

def plot_estimate(estimates, colour, style="-", show_estimates=True):
    plt.plot(estimates.times, estimates.middle, colour, linestyle=style)

    if show_estimates:
        plt.fill_between(estimates.times, estimates.upper, estimates.lower, facecolor=colour, alpha=0.4)


def plot_points(images, colour, marker, fill='none', key=lambda i: i.truth):
    truth = list(map(key, images))
    times = pluck('time', images)

    plt.scatter(times, truth, marker=marker, edgecolors=colour, facecolors=fill)


def pick(images, classes):
    return [i for i in images if i.category in classes]


def plot_point_sets(images, colour):

    plot_points(pick(images, ['train']), colour,   '^')
    plot_points(pick(images, ['validate']), colour, 's')
    plot_points(pick(images, ['discard']), colour,  'o', key=lambda i: i.estimate.middle)

    plot_points(pick(images, ['test']), colour,    'P', fill='r')


def plot_runs(*runs, loc='upper left', show_estimates=True, totals=None, delta=timedelta(minutes=10), size =(20, 10)):
  
    def run_legend(run):
        return Line2D([0], [0], color=run.colour, linestyle=run.get('style', '-'), label=run.label)

    legend = list(map(run_legend, runs)) + ([run_legend(totals)] if totals is not None else []) + [
        # Line2D([0], [0], marker='P', color='r', markeredgecolor='k', linestyle='None', label='test'),
        Line2D([0], [0], marker='^', color='none',  markeredgecolor='k', linestyle='None', label='train'),
        Line2D([0], [0], marker='s', color='none', markeredgecolor='k', linestyle='None', label='validate'),
        Line2D([0], [0], marker='o', color='none', markeredgecolor='k', linestyle='None', label='discard')
    ] 

    fig, ax = make_chart(size = size)

    plt.xlabel("date")
    plt.ylabel("count")

    plt.gcf().autofmt_xdate()

    ax.set_xlim([datetime(2018,12,20), datetime(2019, 2, 28)])
    # s = datetime(2019, 1, 27, 17, 51, 0, tzinfo=tzutc())
    # e = datetime(2019, 1, 29, 21, 13, 0, tzinfo=tzutc())

    # ax.set_xlim([s, e])
    
    estimates = [get_estimates(run.data, delta=delta) for run in runs]

    for run, estimate in zip(runs, estimates):
        plot_estimate(estimate, colour=run.colour, style=run.get('style', '-'), show_estimates=show_estimates)

    if totals is not None:
        total_estimates = sum_estimates(estimates)
        plot_estimate(total_estimates, colour=totals.colour, style=totals.get('style', '-'), show_estimates=show_estimates)

    for run in runs:
        plot_point_sets(run.data, run.colour)

    ax.set_ylim(ymin=0)
    ax.legend(handles=legend, loc=loc)
    return fig

def load(filename):
    return load_dataset(path.join(base_path, filename))

datasets = struct(
    scott_base = 'scott_base.json',
    # scott_base_100 = 'scott_base_100.json',
    # seals      = 'seals.json',
    # seals_102  = 'seals_102.json',
    # seals_shanelle  = 'seals_shanelle.json',
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

def plot_together(figure_path, loaded):


    scott_base = get_counts(loaded['scott_base'])
    scott_base_100 = get_counts(loaded['scott_base_100'])

    images_100 = {image.image_file:image for image in scott_base_100 if image.category != 'new'}

    def hide_duplicate(image):
        return image._extend(category = 'ignore' 
            if (image.image_file in images_100) and (image.category != 'discard')
            else image.category)
            

    scott_base = list(map(hide_duplicate, scott_base))

    cam_b_100  = subset("CamB", scott_base_100)
    cam_c_100  = subset("CamC", scott_base_100)

    cam_b  = subset("CamB", scott_base)
    cam_c  = subset("CamC", scott_base)

    fig = plot_runs(
        struct(data = cam_b_100, colour='tab:olive', style="--", label="camera b (100)"),
        struct(data = cam_b, colour='forestgreen', label="camera b"),

        struct(data = cam_c_100, colour='skyblue', style="--", label="camera c (100)" ),
        struct(data = cam_c, colour='royalblue', label="camera c" ),        
        show_estimates=False
    )

    fig.savefig(path.join(figure_path, "scott_base_combined.pdf"), bbox_inches='tight')


def plot_seals(figure_path, loaded):

    seals = get_counts(loaded['seals'])
    shanelle = get_counts(loaded['seals_shanelle'])


    fig = plot_runs(
        struct(data = shanelle, colour='y', label="$seals_b$"),
        struct(data = seals, colour='g', label="$seals$"),
        estimates=False,
        loc='upper right'
    )

    fig.savefig(path.join(figure_path, "seals_combined.pdf"), bbox_inches='tight')



def plot_scott_base(figure_path, dataset, k):

        cam_a  = subset("CameraA", dataset)
        cam_b  = subset("CameraB", dataset)
        cam_c  = subset("CameraC", dataset)

        fig = plot_runs(
            struct(data = cam_a, colour='r', label="camera a"),
            struct(data = cam_b, colour='g', label="camera b"),
            struct(data = cam_c, colour='b', label="camera c"),
            totals = struct(colour='y', label="total"),
            size=(40, 10)
        )

        fig.savefig(path.join(figure_path, k + ".pdf"), bbox_inches='tight')

        export_counts(path.join(figure_path, k + "_cam_a.csv"), cam_a)
        export_counts(path.join(figure_path, k + "_cam_b.csv"), cam_b)
        export_counts(path.join(figure_path, k + "_cam_c.csv"), cam_c)


def plot_counts(figure_path, loaded):
    # plot_together(figure_path, loaded)
    # plot_seals(figure_path, loaded)

    for k in ['scott_base']:
        scott_base = get_counts(loaded[k])

        cam_b  = subset("CamB", scott_base)
        cam_c  = subset("CamC", scott_base)

        fig = plot_runs(
            struct(data = cam_b, colour='g', label="camera b"),
            struct(data = cam_c, colour='y', label="camera c" ),
            totals = struct(colour='b', label="total")
        )

        fig.savefig(path.join(figure_path, k + ".pdf"), bbox_inches='tight')
        export_counts(path.join(figure_path, k + "_cam_b.csv"), cam_b)
        export_counts(path.join(figure_path, k + "_cam_c.csv"), cam_c)


    # for k in ['seals', 'seals_102', 'seals_shanelle']:
    #     seals_total = get_counts(loaded[k])
    #     seals_pairs = get_counts(loaded[k], class_id = 1)

    #     fig = plot_runs(
    #         struct(data = seals_total, colour='y', label="total"),
    #         struct(data = seals_pairs, colour='c', label="pairs"),

    #         loc='upper right'
    #     )

    #     fig.savefig(path.join(figure_path, k + ".pdf"), bbox_inches='tight')
    #     export_counts(path.join(figure_path, k + ".csv"), seals_total)
    #     export_counts(path.join(figure_path, k + "_pairs.csv"), seals_pairs)   


def show_errors(loaded):

    def statistics(count1, count2):
        diffs =  [count1[k] - count2[k] for k in truth.keys()]
        abs_diffs =  [abs(count1[k] - count2[k]) for k in truth.keys()]
        return struct(mean = np.mean(abs_diffs), std = np.std(diffs), max = np.max(abs_diffs))
        

    # print ("--------" + k + "--------")
    truth = {image.image_file:image.truth
        for image in get_counts(loaded['seals']) if image.category=='test'}

    truth2 = {image.image_file:image.truth
        for image in get_counts(loaded['seals_shanelle']) if image.category=='test'}

    estimate = {image.image_file:image.estimate.middle 
        for image in get_counts(loaded['seals']) if image.category=='test'}

    estimate2 = {image.image_file:image.estimate.middle 
        for image in get_counts(loaded['seals_shanelle']) if image.category=='test'}        

    # [(k, truth[k] - estimate[k], truth[k] - truth2[k]) for k in truth.keys()]

    errors = struct (
        human2_human = statistics(truth, truth2),
        human_estimate = statistics(truth, estimate),
        human2_estimate = statistics(truth2, estimate),

        human_estimate2 = statistics(truth, estimate2),
        human2_estimate2 = statistics(truth2, estimate2),
        estimate_estimate2 = statistics(estimate, estimate2),
    )

    pprint_struct(errors)


    

if __name__ == '__main__':
    figure_path = "/home/oliver/sync/figures/seals/"

    razor = load_dataset("/home/oliver/storage/export/razor.json")
    export_counts(path.join(figure_path, "razor.csv"), get_counts(razor))

    # scott_base = load_dataset("/home/oliver/annotate/working/scott_base.json")
    
    # scott_base = load_dataset("/home/oliver/storage/export/scott_base.json")
    print("loaded")
    # plot_scott_base(figure_path, get_counts(scott_base), "scott_base_new")

    # plot_scott_base(figure_path, get_counts(scott_base), "scott_base_new")

    # loaded = datasets._map(load)
    # plot_counts(loaded)
    # show_errors(loaded)




    # plot_counts(path.join())
