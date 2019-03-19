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
