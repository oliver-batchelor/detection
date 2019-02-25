from scripts.datasets import load_dataset, get_counts
from os import path

from tools import *

import datetime
import random
import matplotlib.pyplot as plt

import numpy as np
import torch


base_path = '/home/oliver/storage/export/'

def differences(xs):
    return [t2 - t1 for t1, t2 in zip (xs, xs[1:])]

def pad(xs, n_before, n_after):
    before = xs.new(n_before).fill_(xs[0])
    after = xs.new(n_after).fill_(xs[-1])
    return  torch.cat([before, xs,  after])


def rolling_window(xs, window=5):
    n_before = window // 2

    xs = pad(xs, n_before, window - n_before - 1)
    return xs.unfold(0, window, 1)

def rolling_diff(xs, window=5):
    means = rolling_window(torch.Tensor(xs), window=window).mean(1)
    return (xs - means).abs()
    

def high_variance(xs, window=5, n = 10):

    xs = torch.Tensor(xs)
    windows = rolling_window(xs, window=window)
    diffs = windows.mean(1) - xs

    return [(i.item(), v.item()) for v, i in zip(*diffs.topk(n))]

def get_clamped(xs):
    n = len(xs) - 1

    def f(i):
        return xs[max(0, min(i, n))]

    return f


def get_window(xs, i, window=5):
    
    x = []
    n_before = window // 2
    n_after = window - n_before - 1

    f = get_clamped(xs)
    return [f(i + d) for d in range(-n_before, n_after + 1)]



def load(filename):
    dataset = load_dataset(path.join(base_path, filename))

    image_counts = get_counts(dataset)

    def subset(text):
        return [count for count in image_counts if text in count.imageFile]


    def plot_estimates(images, window=2):
        images = [image for image in images if image.category in ['validate', 'new']]

        estimates = transpose_structs(pluck('estimate', images))
        times = pluck('time', images)
        # files = pluck('imageFile', images)
        for i, v in high_variance(estimates.middle, window=window, n = 20):
            print(images[i].imageFile, v)

            list(map(lambda i: print(i._subset('imageFile', 'estimate')), get_window(images, i, window=window)))
            
    cam_c  = subset("CamC")
    cam_b  = subset("CamB")

    plot_estimates(cam_b)



    # pl.fill_between(x, y-error, y+error)

    # def plot_counts(counts):
        


    # def plot_validation(counts):

    # plt.scatter(, y)

    plt.gcf().autofmt_xdate()



datasets = struct(
    scott_base = 'scott_base.json',
)

if __name__ == '__main__':
         

    loaded = datasets._map(load)

    # plot_counts(path.join())