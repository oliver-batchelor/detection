import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
from tools import struct

from os import path

def area(box):
    lx, ly = box.lower
    ux, uy = box.upper

    return (max(0, ux - lx)) * (max(0, uy - ly))


def iou_box(box1, box2):
    overlap = struct(lower = np.maximum(box1.lower, box2.lower),  
    upper = np.minimum(box1.upper, box2.upper))

    i = area(overlap)
    u = (area(box1) + area(box2) - i)
  
    return i / u


def to_points(box):
  return struct(lower = box.centre - box.size/2, upper = box.centre + box.size/2)


def add_noise(box, offset=0, noise=0):
  centre = box.centre + offset * box.size + np.random.normal(0, noise, 2) * box.size
  size = box.size * np.random.normal(1, noise, 2)

  noisy = struct(centre = centre, size = size)
        
  iou = iou_box(to_points(noisy), to_points(box))
  return noisy, iou

def average_iou(offset=0, noise=0, samples=100000):

  box = struct(centre = np.array([0.0, 0]), size = np.array([1.0, 1]))
  ious = [add_noise(box, offset / 100, noise / 100)[1] for i in range(0, samples)]

  return sum(ious) / samples

def plot_boxes(ax):

  def add_box(box, color='g', linewidth=1):
    rect = patches.Rectangle(box.centre - box.size / 2, *box.size, linewidth=linewidth,edgecolor=color,facecolor='none')
    ax.add_patch(rect)

  def magnitude(i):
    return 0 if i == 0 else pow(2, i + 1)

  plt.sca(ax)
  colors = plt.get_cmap("tab10")


  for i in range(0, 5):
    for j in range(0, 5):
      noise = magnitude(i)
      offset = magnitude(j)
  
      box = struct(centre = np.array([i+1, j+1]), size = np.array([0.5,0.5]))
      add_box(box)



      for k in range(0, 10):
        noisy, _ = add_noise(box, offset = offset / 100, noise = noise / 100)
        add_box(noisy, color=colors(k), linewidth=0.5)

      miou = average_iou(offset, noise)
      ax.text(box.centre[0], box.centre[1], "{:.1f}".format(miou * 100), horizontalalignment='center', verticalalignment='center')



  ax.set_xlim(xmin=0, xmax=6)  
  ax.set_ylim(ymin=0, ymax=6)  

  ax.set_xlabel('noise level (center and size) $\sigma$ as percent of box size')
  ax.set_ylabel('systematic offset $\Delta$ as percent of box size')

  plt.xticks([1,2,3,4,5], ['0', '4', '8', '16', '32'])
  plt.yticks([1,2,3,4,5], ['0', '4', '8', '16', '32'])

  
def plot_noisy_boxes():

  fig, ax = plt.subplots(1, 1, figsize=(8, 8), gridspec_kw = {'wspace':0.1, 'hspace':0.1})  
  plot_boxes(ax)

  return fig, ax


if __name__=='__main__':
  figure_path = "/home/oliver/sync/figures/training"

  fig, ax = plot_noisy_boxes()
  fig.savefig(path.join(figure_path, "noisy_boxes.pdf"), bbox_inches='tight')


  