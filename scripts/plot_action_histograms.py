from scripts.load_figures import *
from matplotlib.lines import Line2D

from scripts.history import *
import scripts.figures

from tools import transpose_structs, struct

from scipy import stats
import numpy as np




def make_splits(xs, n_splits=None):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
 
def get_corrections(image):
    return image_summary(image).correction_count

def get_actions(image):
    return image_summary(image).actions_count


def basic_histograms(values, keys):
    n = len(actions)
    
    fig, ax = make_chart()
    plot_stacks(np.array(range(n)) + 0.5, actions, keys, width= 0.5)
    return fig, ax


def uneven_histograms(widths, values, keys):
    times = (np.cumsum([0] + widths[:-1])) + np.array(widths) * 0.5
   
    fig, ax = make_chart()
    plot_stacks(times, values, keys, width=widths)

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0) 

    return fig, ax


def uneven_gaussian_filter(x, y, w=1, dx=0.1, sigma=1):
    x_eval = np.arange(0, np.amax(x), dx)

    delta_x = x_eval[:, None] - x

    weights = w * np.exp(-delta_x*delta_x / (2*sigma*sigma)) / (np.sqrt(2*np.pi) * sigma)
    weights /= np.sum(weights, axis=1, keepdims=True)
    y_eval = np.dot(weights, y)

    return x_eval, y_eval

annotation_types = correction_types[:]
annotation_types.remove('false positive')


def get_annotation_counts(dataset):
    counts = [image_summary(image).correction_count for image in dataset.history]
    return transpose_structs(counts)._map(np.array)._subset(*annotation_types)

def get_action_counts(dataset):
    counts = [image_summary(image).actions_count for image in dataset.history]
    return transpose_structs(counts)._map(np.array)

def get_ratios(counts_struct):
    total = sum(counts_struct.values())
    denom = np.maximum(total, 1)
    return counts_struct._map(lambda c: c/denom), total

def get_time(dataset):
    durations = np.array(pluck('duration', dataset.history)) / 60
    time = np.cumsum(durations)
    return time, durations

def get_normalised_time(dataset):
    durations = np.array(pluck('duration', dataset.history)) / 60
    time = np.cumsum(durations)
    return time/time[-1], durations


def plot_annotation_stack(ax, dataset, sigma=5):
    ratios, total = get_ratios(get_annotation_counts(dataset))
    time, _ = get_time(dataset)

    x, y = zip(*[uneven_gaussian_filter(time, ratios[k], total, sigma=sigma) 
        for k in annotation_types])
    
    colors = [correction_colors[k] for k in annotation_types]
    ax.stackplot(x[0], *y, colors=colors, labels=annotation_types, alpha=0.8)

def plot_action_stack(ax, dataset, sigma=5):
    ratios, total = get_ratios(get_action_counts(dataset))
    
    time, _ = get_time(dataset)
    x, y = zip(*[uneven_gaussian_filter(time, ratios[k], total, sigma=sigma) 
        for k in action_types])
    
    colors = [action_colors[k] for k in action_types]
    ax.stackplot(x[0], *y, colors=colors, labels=action_types, alpha=0.8)


def plot_instance_lines(ax, dataset, sigma=5):
    annotation_counts = get_annotation_counts(dataset)
    total = sum(annotation_counts.values())

    time, durations = get_time(dataset)

    x, y = uneven_gaussian_filter(time, total / durations, 
        durations, sigma=sigma)

    ax.plot(x, y, label='total')
    ax.grid(True)
    

def plot_combined_ratios(dataset, sigma=5):
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(16, 12))  
    fig.subplots_adjust(hspace=0)

    time, _ = get_time(dataset)

    plot_annotation_stack(ax[0], dataset, sigma=sigma)
    plot_action_stack(ax[1], dataset, sigma=sigma)
    plot_instance_lines(ax[2], dataset, sigma=sigma)

    ax[0].set_xlim(xmin=0, xmax=time[-1]) 
    ax[0].set_ylim(ymin=0, ymax=1)
    ax[1].set_ylim(ymin=0, ymax=1)

    ax[1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ticks = ax[2].get_yticks()
    ax[2].set_yticks(ticks[:-1])

    ax[0].set_ylabel('proportion')
    ax[1].set_ylabel('proportion')
    ax[2].set_ylabel('annotation rate')

    plt.xlabel('annotation time (minutes)')
	
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    return fig, ax



def plot_annotation_ratios(dataset, sigma=5):
    fig, ax = make_chart(grid=False)

    plot_annotation_stack(ax, dataset, sigma=sigma)
    plot_action_stack(ax, dataset, sigma=sigma)


    ax.set_ylim(ymin=0, ymax=1)
    ax.set_xlim(xmin=0) 

    plt.xlabel('annotation time (minutes)')
    plt.ylabel('proportion of annotations')

    plt.legend()

    return fig, ax

def plot_action_ratios(dataset, sigma=5):
    fig, ax = make_chart(grid=False)

    plot_action_stack(ax, dataset, sigma=sigma)

    ax.set_ylim(ymin=0, ymax=1)
    ax.set_xlim(xmin=0) 

    plt.xlabel('annotation time (minutes)')
    plt.ylabel('proportion of actions')

    plt.legend()

    return fig, ax    

def plot_instance_rates(datasets, color_map, labels, sigma=5):
    fig, ax = make_chart()

    for k, dataset in datasets.items():
        annotation_counts = get_annotation_counts(dataset)
        total = sum(annotation_counts.values())

        time, durations = get_time(dataset)

        x, y = uneven_gaussian_filter(time, total / durations, 
             durations, sigma=sigma)

        plt.plot(100 * (x / time[-1]), y, color=color_map[k], label=labels[k])

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0, xmax=100) 

    plt.xlabel('annotation time (percent)')
    plt.ylabel('annotation rate')


    plt.legend()
    return fig, ax

    

def plot_dataset_ratios(datasets, color_map, labels, sigma=5):
    fig, ax = make_chart()
   
    for k, dataset in datasets.items():
        ratios, total = get_ratios(get_annotation_counts(dataset))

        time, _ = get_time(dataset)
        x, y = uneven_gaussian_filter(time, ratios.positive, total, sigma=sigma)

        plt.plot(100 * (x / time[-1]), y, color=color_map[k], label=labels[k])

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0, xmax=100) 

    plt.xlabel('annotation time (percent)')
    plt.ylabel('proportion of annotations')


    plt.legend()

    return fig, ax



def cumulative_lines(dataset, get_values, keys):

    actions = [get_values(image) for image in dataset.history]        
    durations = torch.Tensor(pluck('duration', dataset.history)).cumsum(0)

    fig, ax = make_chart()
    plot_cumulative_line_stacks(durations, actions, keys)
    plt.show()    


def cumulative_instances(datasets, color_map, labels):

    fig, ax = make_chart()

    for k in sorted(datasets.keys()):
        dataset = datasets[k]
        
        summaries = image_summaries(dataset.history)

        instances = torch.Tensor([0] + pluck('instances', summaries)).cumsum(0)
        durations = torch.Tensor([0] + pluck('duration', summaries)).cumsum(0)

        plt.plot((durations / 60).numpy(), instances.numpy(), label = labels[k])

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)


    ax.set_xlabel("annotation time (m)")
    ax.set_ylabel("count")


    ax.legend()
    
    return fig, ax    


def thresholds(image):
    changes = [action.value for action in image.actions if action.action=='threshold']
    return [image.threshold] + changes


def plot_action_histograms(loaded, color_map, labels, figure_path):
    fig, ax = plot_instance_rates(loaded, color_map, labels=labels, sigma=5)
    fig.savefig(path.join(figure_path, "summaries/instance_rates.pdf"), bbox_inches='tight')

    fig, ax = plot_dataset_ratios(loaded, color_map, labels=labels, sigma=5)
    fig.savefig(path.join(figure_path, "summaries/positive_ratio.pdf"), bbox_inches='tight')

    for k, dataset in loaded.items():
        fig, ax = plot_combined_ratios(loaded[k], sigma=5)
        fig.savefig(path.join(figure_path, "action_annotations", k + ".pdf"), bbox_inches='tight')        

    fig, ax = cumulative_instances(loaded, color_map, labels=labels)
    fig.savefig(path.join(figure_path, "summaries/cumulative_instances.pdf"), bbox_inches='tight')

    ax.set_ylim(ymin=0, ymax=8000)
    ax.set_xlim(xmin=0, xmax=150)
    fig.savefig(path.join(figure_path, "summaries/cumulative_instances_crop.pdf"), bbox_inches='tight')



if __name__ == '__main__':
    figure_path = "/home/oliver/sync/figures"

    loaded = load_all(datasets._without('aerial_penguins'), base_path)

    summaries = pluck_struct('summary', loaded)
    pprint_struct(summaries)

    plot_action_histograms(loaded, color_map=dataset_colors, labels=dataset_labels, figure_path=figure_path)



