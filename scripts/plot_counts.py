from scripts.datasets import load_dataset, get_counts
from os import path

from tools import *
import tools.window as window


import datetime
import random
import matplotlib.pyplot as plt

import numpy as np
import torch


base_path = '/home/oliver/storage/export/'


def smooth(size=11):
    def f(xs):
        return window.rolling_window(torch.Tensor(xs), window=size).mean(1).numpy()
    return f


def load(filename):
    dataset = load_dataset(path.join(base_path, filename))

    image_counts = get_counts(dataset)

    def subset(text):
        return [count for count in image_counts if text in count.image_file]

    def plot_estimate(images, colour, label):
        estimates = transpose_structs(pluck('estimate', images))
        times = pluck('time', images)

        # middle = window.rolling_window(torch.Tensor(estimates.middle), window=5).mean(1).numpy()
        estimates = estimates._map(smooth(5))
      
        
        plt.plot(times, estimates.middle, colour, label='estimate ' + label)
        plt.fill_between(times, estimates.upper, estimates.lower, facecolor=colour, alpha=0.4)

    def plot_truth(images, marker, label):
        truth = pluck('truth', images)
        times = pluck('time', images)

        plt.plot(times, truth, marker, label=label)


    def plot_subset(images, colour, label):

        images = [i for i in images if i.category != 'discard']
        plot_estimate(images, colour, label)

        train = [i for i in images if i.category == 'train']
        plot_truth(train, colour + '+', label = 'train ' + label)

        validate = [i for i in images if i.category == 'validate']
        plot_truth(validate, colour + '.', label = 'validate ' + label)


            
    cam_c  = subset("CamC")
    cam_b  = subset("CamB")

    plt.xlabel("Date")
    plt.ylabel("Count")

    plt.gcf().autofmt_xdate()

    
    plot_subset(cam_b, 'y', 'b')
    plot_subset(cam_c, 'g', 'c')


    plt.legend(loc='upper left')

    plt.show()



    # pl.fill_between(x, y-error, y+error)

    # def plot_counts(counts):
        


    # def plot_validation(counts):

    # plt.scatter(, y)

    



datasets = struct(
    scott_base = 'scott_base.json',
)

if __name__ == '__main__':
         

    loaded = datasets._map(load)

    # plot_counts(path.join())