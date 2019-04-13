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


if __name__ == '__main__':

    loaded = load_all(datasets, base_path)
    keys = sorted(loaded.keys())

    pprint_struct(pluck_struct('summary', loaded))

    instances_duration(loaded)
        
    

 

    plt.show()
    # fig.savefig(path.join(figure_path, "cum_instances.pdf"), bbox_inches='tight')


