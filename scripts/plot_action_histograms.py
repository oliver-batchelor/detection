from scripts.make_figures import *
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
    
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_stacks(np.array(range(n)) + 0.5, actions, keys, width= 0.5)
    return fig, ax


def uneven_histograms(widths, values, keys):
    times = (np.cumsum([0] + widths[:-1])) + np.array(widths) * 0.5
   
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_stacks(times, values, keys, width=widths)

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0) 

    return fig, ax


def uneven_gaussian_filter(x, y, w=1, dx=0.1, sigma=1):
    x_eval = np.arange(np.amin(x), np.amax(x), dx)

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


def plot_annotation_ratios(dataset, sigma=5):
    fig, ax = plt.subplots(figsize=(24, 12))

    ratios, total = get_ratios(get_annotation_counts(dataset))
    
    time, _ = get_time(dataset)

    x, y = zip(*[uneven_gaussian_filter(time, ratios[k], total, sigma=sigma) 
        for k in annotation_types])
    
    colors = [correction_colors[k] for k in annotation_types]
    plt.stackplot(x[0], *y, colors=colors, labels=annotation_types)

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0) 

    plt.xlabel('annotation time (minutes)')
    plt.ylabel('proportion of annotations')

    plt.title('proportions of corrections (and type)')
    plt.legend()

    return fig, ax

def plot_action_ratios(dataset, sigma=5):
    fig, ax = plt.subplots(figsize=(24, 12))

    ratios, total = get_ratios(get_action_counts(dataset))
    
    time, _ = get_time(dataset)
    x, y = zip(*[uneven_gaussian_filter(time, ratios[k], total, sigma=sigma) 
        for k in action_types])
    
    colors = [action_colors[k] for k in action_types]
    plt.stackplot(x[0], *y, colors=colors, labels=action_types)

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0) 

    plt.xlabel('annotation time (minutes)')
    plt.ylabel('proportion of actions')

    plt.title('user actions over time')
    plt.legend()

    return fig, ax    

def plot_instance_rates(datasets, sigma=5):
    fig, ax = plt.subplots(figsize=(24, 12))

    for k, dataset in datasets.items():
        annotation_counts = get_annotation_counts(dataset)
        total = sum(annotation_counts.values())

        time, durations = get_time(dataset)

        x, y = uneven_gaussian_filter(time, total / durations, 
             durations, sigma=sigma)

        plt.plot(x / time[-1], y, color=dataset_colors[k], label=k)

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0) 

    plt.xlabel('annotation time (percent)')
    plt.ylabel('annotation rate (counts/minute)')

    plt.title('annotation rate across annotation period')

    plt.legend()
    return fig, ax

    

def plot_dataset_ratios(datasets, sigma=5):
    fig, ax = plt.subplots(figsize=(24, 12))
   
    for k, dataset in datasets.items():
        ratios, total = get_ratios(get_annotation_counts(dataset))

        time, _ = get_time(dataset)
        x, y = uneven_gaussian_filter(time, ratios.positive, total, sigma=sigma)

        plt.plot(x / time[-1], y, color=dataset_colors[k], label=k)

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0) 

    plt.xlabel('annotation time (percent)')
    plt.ylabel('proportion of annotations')

    plt.legend()

    return fig, ax



def cumulative_lines(dataset, get_values, keys):

    actions = [get_values(image) for image in dataset.history]        
    durations = torch.Tensor(pluck('duration', dataset.history)).cumsum(0)

    fig, ax = plt.subplots(figsize=(24, 12))
    plot_cumulative_line_stacks(durations, actions, keys)
    plt.show()    


def thresholds(image):
    changes = [action.value for action in image.actions if action.action=='threshold']
    return [image.threshold] + changes

if __name__ == '__main__':
    figure_path = "/home/oliver/sync/figures"

    loaded = load_all(datasets, base_path)._without('aerial_penguins')

    summaries = pluck_struct('summary', loaded)
    pprint_struct(summaries)

    fig, ax = plot_instance_rates(loaded, sigma=5)
    fig.savefig(path.join(figure_path, "summaries/instance_rates.pdf"), bbox_inches='tight')

    fig, ax = plot_dataset_ratios(loaded, sigma=5)
    fig.savefig(path.join(figure_path, "summaries/positive_ratio.pdf"), bbox_inches='tight')

    for k, dataset in loaded.items():
        fig, ax = plot_annotation_ratios(loaded[k], sigma=5)
        plt.title(k + ' - annotation proportions vs. annotation time')
        fig.savefig(path.join(figure_path, "annotation_ratio", k + ".pdf"), bbox_inches='tight')

        fig, ax = plot_action_ratios(loaded[k], sigma=5)
        plt.title(k + ' - action proportions vs. annotation time')
        fig.savefig(path.join(figure_path, "action_ratio", k + ".pdf"), bbox_inches='tight')

