from scripts.load_figures import *
from matplotlib.lines import Line2D

from scripts.datasets import *
from tools import to_structs, transpose_structs, transpose_dicts

import scripts.figures

from dataset.imports.voc import import_voc
from scipy import stats

from scripts.figures import *
import csv


def plot_sizes(loaded, keys, labels=dataset_labels):

    length_quartiles = {k : loaded[k].summary.box_length.quartiles for k in keys}
    fig, ax = box_plot(length_quartiles, keys, labels=dataset_labels)

    ax.set_xscale('log')
    ax.set_xlim(xmin=10, xmax=1e3)

    ax.set_xlabel('largest side length (pixels)')  
    return fig, ax

def plot_durations(loaded, keys, color_map=dataset_colors, labels=dataset_labels):   
    durations_quartiles = {k : loaded[k].summary.image_durations.quartiles  for k in keys}
    fig, ax = box_plot(durations_quartiles, keys, labels=labels)

    for i, k in enumerate(keys):
        counts = np.array(loaded[k].image_summaries.instances)
        durations = np.array(loaded[k].image_summaries.duration)



        plt.scatter(durations, np.repeat(i, durations.shape), marker = '|', \
             s=np.clip(counts, a_min=1, a_max=None) * 20, color=color_map[k])    

    ax.set_xscale('log')
    # ax.set_xlim(xmin=1)

    ax.set_xlabel('image annotation time (s)')  
    return fig, ax




def plot_instances_density(loaded, keys, color_map=dataset_colors, labels=dataset_labels):
    instances_quartiles = {k : loaded[k].summary.instances_image.quartiles + 1 for k in keys}

    fig, ax = box_plot(instances_quartiles, keys, labels=labels)

    for i, k in enumerate(keys):
        counts = np.array(loaded[k].image_summaries.instances)

        density = stats.kde.gaussian_kde(counts,  bw_method=0.2)

        x = np.logspace(counts.min(), counts.max(), 400)
        d = density(x)

        ax.fill_between(x, i + d, i - d, color=color_map[k])
    
    ax.set_xscale('log')
    ax.set_xlim(xmin=0.95, xmax=1e3)

    ax.set_xlabel('annotations per image (+1)')  
    return fig, ax


def plot_instances(loaded, keys, color_map=dataset_colors, labels=dataset_labels):
    instances_quartiles = {k : loaded[k].summary.instances_image.quartiles + 1 for k in keys}

    fig, ax = box_plot(instances_quartiles, keys, labels=labels)

    for i, k in enumerate(keys):
        counts = np.array(loaded[k].image_summaries.instances)

        counts = np.bincount(counts)

        instances = np.nonzero(counts)[0]
        counts = counts[instances]

        plt.scatter(instances + 1, np.repeat(i, instances.shape), s = 100, marker='|', color=color_map[k])
    
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


def plot_category_bars(stacks, keys, color_map, categories, cat_labels):
    fig, ax = make_chart(size=(8, 6))

    n = len(stacks)
    width = 0.8 / len(keys)

    bars = []
    for i, k in enumerate(keys):
        values = np.array([stacks[c][k] for c in categories])

        p = ax.bar(np.arange(n) - 0.4 + (i + 0.5) * width, values, width, color=color_map[k])
        bars.append(p[0])
    
    plt.xticks(np.arange(n), [cat_labels[c] for c in categories], rotation='vertical')
    plt.legend(bars, keys)
    return fig, ax





def plot_times_density(loaded, keys, color_map, labels):
    fig, ax = make_chart()

    for k in keys:
        summaries = image_summaries(loaded[k].history)
        times = [action.real_duration for action in sum_list(summaries).actions]

        density = stats.kde.gaussian_kde(times,  bw_method=0.1)
   
        x = np.arange(0.0, 80, .1)
        plt.plot(x, density(x), label = labels[k], color = color_map[k])

    plt.xlabel('time taken(s)')
    plt.ylabel('density')

    plt.legend()

    ax.set_xlim(xmin=0.0, xmax=80.0)
    ax.set_ylim(ymin=0.0)
    return fig, ax


def export_summary_table(filename, keys, summaries, labels):

    def get_entry(e):
        if type(e) is Struct and 'mean' in e:
          return "${:.3g} \pm {:.3g}$".format(e.mean, e.std)
        elif type(e) is float:
          return "{:.3g}".format(e)
        else:
          return str(e)

    def entries(k, summary):
      return summary._subset(*keys)._map(get_entry)._extend(name = labels[k])

    rows = list(summaries._mapWithKey(entries).values())
    export_csv(filename,  ['name'] + keys, rows)


def export_summary_csvs(figure_path, summaries, labels):

    data_keys = ['n_annotations', 'n_images', 'box_length', 'size_ranges'] 
    export_summary_table(path.join(figure_path, "data_summary.csv"), data_keys,  summaries, labels)

    annotation_keys = ['n_actions', 'n_annotations', 'total_minutes', 'instances_minute', 'actions_minute', 'actions_annotation']
    
    export_summary_table(path.join(figure_path, "time_summary.csv"), annotation_keys,  summaries, labels)

    export_summary_table(path.join(figure_path, "correction_summary.csv"), correction_types, summaries._map(lambda s: s.correction_count), labels) 


if __name__ == '__main__':
    figure_path = "/home/oliver/sync/figures/summaries"

 
    loaded = load_all(datasets, base_path)
    summaries = pluck_struct('summary', loaded)
    pprint_struct(summaries)

    export_summary_csvs(figure_path, summaries, dataset_labels)

    loaded = loaded._without('seals2')
    keys=sorted(loaded.keys())
    summaries = pluck_struct('summary', loaded)

    fig, ax = plot_times_density(loaded, keys, color_map=dataset_colors, labels=dataset_labels)
    fig.savefig(path.join(figure_path, "time_density.pdf"), bbox_inches='tight')
 

    actions_proportions = summaries._map(lambda s: s.actions_count / s.total_actions)
    fig, ax = plot_category_bars( actions_proportions, action_types, color_map=action_colors, categories=keys, cat_labels=dataset_labels)
    ax.set_ylabel('proportion of actions')
    fig.set_size_inches(8, 7)
    fig.savefig(path.join(figure_path, "action_counts.pdf"), bbox_inches='tight')

    correction_types = ['weak positive', 'modified positive', 'false negative', 'false positive']
    
    correction_proportions = summaries._map(lambda s: s.correction_count / s.n_annotations)
    fig, ax = plot_category_bars(correction_proportions,  correction_types, color_map=correction_colors, categories=keys, cat_labels=dataset_labels)
    ax.set_ylabel('proportion of annotation count')

    ax.set_ylim(ymax=0.15)
    fig.set_size_inches(8, 7)
    fig.savefig(path.join(figure_path, "correction_counts.pdf"), bbox_inches='tight')

    fig, ax = plot_instances(loaded, keys=keys)
    fig.set_size_inches(16, 9)
    fig.savefig(path.join(figure_path, "instances_boxplot.pdf"), bbox_inches='tight')

    fig, ax = plot_durations(loaded, keys=keys)
    fig.set_size_inches(16, 9)
    fig.savefig(path.join(figure_path, "duration_boxplot.pdf"), bbox_inches='tight')

    fig, ax = plot_sizes(loaded, keys=keys)
    fig.set_size_inches(16, 9)
    fig.savefig(path.join(figure_path, "sizes_boxplot.pdf"), bbox_inches='tight')
