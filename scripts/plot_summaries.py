from scripts.load_figures import *
from matplotlib.lines import Line2D

from scripts.datasets import *
from tools import to_structs

import scripts.figures

from dataset.imports.voc import import_voc
from scipy import stats

from scripts.figures import *


def plot_sizes(loaded, keys):

    length_quartiles = {k : loaded[k].summary.box_length for k in keys}
    fig, ax = box_plot(length_quartiles, keys)

    ax.set_xscale('log')
    ax.set_xlim(xmin=10, xmax=1e3)

    ax.set_xlabel('largest side length (pixels)')  
    return fig, ax

def plot_durations(loaded, keys, color_map=dataset_colors):
    durations_quartiles = {k : loaded[k].summary.image_durations  for k in keys}

    fig, ax = box_plot(durations_quartiles, keys)

    for i, k in enumerate(keys):
        counts = np.array(pluck('instances', loaded[k].image_summaries))

        durations = np.array(pluck('duration', loaded[k].image_summaries))
        plt.scatter(durations, np.repeat(i, durations.shape), marker = '|', \
             s=np.clip(counts, a_min=1, a_max=None) * 20, color=color_map[k])    

    ax.set_xscale('log')
    # ax.set_xlim(xmin=1)

    ax.set_xlabel('image annotation time (s)')  
    return fig, ax



def plot_instances(loaded, keys, color_map=dataset_colors):
    instances_quartiles = {k : loaded[k].summary.instances_image + 1 for k in keys}

    fig, ax = box_plot(instances_quartiles, keys)

    for i, k in enumerate(keys):
        counts = np.bincount(pluck('instances', loaded[k].image_summaries))

        instances = np.nonzero(counts)[0]
        counts = counts[instances]

        plt.scatter(instances + 1, np.repeat(i, instances.shape), s = 50, linewidths=counts, \
             marker = '|', color=color_map[k])
    
    ax.set_xscale('log')
    ax.set_xlim(xmin=0.95, xmax=1e3)

    ax.set_xlabel('annotations per image (+1)')  
    return fig, ax

def plot_category_stacks(stacks, keys, color_map, categories):
    fig, ax = make_chart()

    total = np.array([0] * len(categories))

    n = len(stacks)

    bars = []
    for i, k in enumerate(keys):
        values = np.array([stacks[c][k] for c in categories])

        p = ax.barh(np.arange(n) + 0.5, values, 0.5, left=total, color=color_map[k])
        bars.append(p[0])

        total = total + values
    
    plt.yticks(np.arange(n) + 0.5, categories)
    plt.legend(bars, keys)
    return fig, ax


def plot_category_bars(stacks, keys, color_map, categories):
    fig, ax = make_chart()

    n = len(stacks)
    width = 0.8 / len(keys)

    bars = []
    for i, k in enumerate(keys):
        values = np.array([stacks[c][k] for c in categories])

        p = ax.barh(np.arange(n) - 0.4 + (i + 0.5) * width, values, width, color=color_map[k])
        bars.append(p[0])
    
    plt.yticks(np.arange(n), categories)
    plt.legend(bars, keys)
    return fig, ax




if __name__ == '__main__':
    figure_path = "/home/oliver/sync/figures/summaries"

 
    loaded = load_all(datasets, base_path)
    keys = sorted(loaded.keys())
   
    summaries = pluck_struct('summary', loaded)
    pprint_struct(summaries)
  
    fig, ax = plot_category_bars( pluck_struct('actions_count', summaries), action_types, color_map=action_colors, categories=keys)
    ax.set_title('total action counts for each dataset')
    ax.set_xlabel('action count')
    ax.set_xlim(xmax=4000)
    fig.savefig(path.join(figure_path, "action_counts.pdf"), bbox_inches='tight')

    correction_types = ['weak positive', 'modified positive', 'false negative', 'false positive']
    fig, ax = plot_category_bars( pluck_struct('correction_count', summaries), correction_types, color_map=correction_colors, categories=keys)
    ax.set_xlabel('correction count')
    ax.set_title('total correction types performed for each dataset')
    fig.savefig(path.join(figure_path, "correction_counts.pdf"), bbox_inches='tight')

    fig, ax = plot_instances(loaded, keys=keys)
    fig.savefig(path.join(figure_path, "instances_boxplot.pdf"), bbox_inches='tight')

    fig, ax = plot_durations(loaded, keys=keys)
    fig.savefig(path.join(figure_path, "duration_boxplot.pdf"), bbox_inches='tight')

    fig, ax = plot_sizes(loaded, keys=keys)
    fig.savefig(path.join(figure_path, "sizes_boxplot.pdf"), bbox_inches='tight')
