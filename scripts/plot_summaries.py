from scripts.make_figures import *
from matplotlib.lines import Line2D

import scripts.figures

figure_path = "/home/oliver/sync/figures/summaries"


def plot_sizes(loaded):

    length_quartiles = {k : loaded[k].summary.box_length for k in loaded.keys()}
    fig, ax = box_plot(length_quartiles, keys, colour_map)

    ax.set_xscale('log')
    ax.set_xlim(xmin=10, xmax=1e3)

    ax.set_xlabel('largest side length (pixels)')  
    return fig, ax

def plot_durations(loaded):
    durations_quartiles = {k : loaded[k].summary.image_durations  for k in loaded.keys()}

    fig, ax = box_plot(durations_quartiles, keys, colour_map)

    for i, k in enumerate(keys):
        durations = np.array(pluck('duration', loaded[k].image_summaries))
        plt.scatter(durations, np.repeat(i, durations.shape), marker = '|', s=100, color=colour_map[k])    

    ax.set_xscale('log')
    ax.set_xlim(xmin=0.1)

    ax.set_xlabel('image annotation time (s)')  
    return fig, ax



def plot_instances(loaded):
    instances_quartiles = {k : loaded[k].summary.instances_image + 1 for k in loaded.keys()}

    fig, ax = box_plot(instances_quartiles, keys)

    for i, k in enumerate(keys):
        counts = np.bincount(pluck('instances', loaded[k].image_summaries))

        instances = np.nonzero(counts)[0]
        counts = counts[instances]

        plt.scatter(instances + 1, np.repeat(i, instances.shape), s=counts * 20, \
             marker = '|', color=colour_map[k])
    
    ax.set_xscale('log')
    ax.set_xlim(xmin=0.95, xmax=1e3)

    ax.set_xlabel('annotations per image (+1)')  
    return fig, ax

def plot_category_stacks(stacks, keys):
    fig, ax = plt.subplots(figsize=(24, 12))

    categories = sorted(stacks.keys())
    total = np.array([0] * len(categories))

    n = len(stacks)

    bars = []
    for i, k in enumerate(keys):
        values = np.array([stacks[c][k] for c in categories])

        p = ax.barh(np.arange(n) + 0.5, values, 0.5, left=total)
        bars.append(p[0])

        total = total + values
    
    plt.yticks(np.arange(n) + 0.5, categories)
    plt.legend(bars, keys)
    return fig, ax


def plot_category_bars(stacks, keys):
    fig, ax = plt.subplots(figsize=(24, 12))

    categories = sorted(stacks.keys())
    n = len(stacks)
    width = 0.8 / len(keys)

    bars = []
    for i, k in enumerate(keys):
        values = np.array([stacks[c][k] for c in categories])

        p = ax.barh(np.arange(n) - 0.4 + (i + 0.5) * width, values, width)
        bars.append(p[0])

    
    plt.yticks(np.arange(n), categories)
    plt.legend(bars, keys)
    return fig, ax




if __name__ == '__main__':

    loaded = load_all(datasets._subset('penguins', 'branches'), base_path)
    keys = sorted(loaded.keys())

    summaries = pluck_struct('summary', loaded)
    pprint_struct(summaries)
  
    fig, ax = plot_category_bars( pluck_struct('actions_count', summaries), action_types)
    ax.set_title('total action counts for each dataset')
    ax.set_xlabel('action count')
    fig.savefig(path.join(figure_path, "action_counts.pdf"), bbox_inches='tight')


    annotation_types = ['weak positive', 'modified positive', 'false negative', 'false positive']
    fig, ax = plot_category_bars( pluck_struct('annotation_types', summaries), annotation_types)
    ax.set_xlabel('correction count')
    ax.set_title('total correction types performed for each dataset')
    fig.savefig(path.join(figure_path, "correction_counts.pdf"), bbox_inches='tight')



    # plt.show()    

    

    # fig, ax = instances_duration_scatter(loaded, ['apples1', 'apples2'])
    # ax.set_ylim(ymin=0)
    # ax.set_xlim(xmin=0)      
    # fig.savefig(path.join(figure_path, "instances_duration_scatter_apples.pdf"), bbox_inches='tight')



