from scripts.load_figures import *
from matplotlib.lines import Line2D

from scripts.history import *
import scripts.figures

from tools import transpose_structs, struct

from scipy import stats
import numpy as np



def image_result(image):      
    prediction = empty_detections if image.detections is None else image.detections
    return struct(target = image.target, prediction = prediction )

def running_AP(dataset, dx=0.1, sigma=5):
    ious = [t for t in list(range(50, 100, 5))]
    times, mAPs = running_mAP(dataset, iou=ious, dx=dx, sigma=sigma)
    return times, sum(mAPs.values()) / len(mAPs), mAPs

def running_mAP(dataset, iou=[50], dx=0.1, sigma=5):

    duration = torch.Tensor(pluck('duration', dataset.history))
    times = (duration / 60).cumsum(0)
    eval_times = torch.arange(torch.min(times), torch.max(times), dx)
   
    image_pairs =  filter_none([image_result(image) for image in dataset.history])
    mAP = evaluate.mAP_smoothed(image_pairs, times) 

    return eval_times.numpy(), { t : mAP(t/100.0, eval_times, sigma).numpy() for t in iou }


def compute_mAPs(datasets, iou=[50], sigma=5):
    def f(dataset):
        dx = dataset.summary.total_minutes / 400
        times, mAPs = running_mAP(dataset, iou=iou, dx=dx, sigma=sigma)   
        return struct(times=times, mAPs=mAPs)
    return datasets._map(f)


def compute_APs(datasets, sigma=5):
    def f(dataset):
        dx = dataset.summary.total_minutes / 400
        times, AP, mAPs = running_AP(dataset, dx=dx, sigma=sigma)   
        return struct(times=times, AP=AP, mAPs=mAPs)

    return datasets._map(f)

def plot_running_mAPs(results, ious = [50, 75, 90], color_map=dataset_colors):
    fig, ax = plt.subplots(figsize=(24, 12))

    for k, r in results.items():
        
        styles = ['-', '--', '-.', ':']
        for iou, style in zip(ious, styles):
            plt.plot(r.times / r.times[-1], r.mAPs[iou], linestyle=style, 
                 color=color_map[k], label=k + " mAP@." + str(iou))
            

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0) 

    plt.xlabel('annotation time (percent)')
    plt.ylabel('average precision')

    plt.title('running mAP of predictions vs. corrected images')

    plt.legend()
    return fig, ax


def plot_APs_dataset(r):
    fig, ax = plt.subplots(figsize=(24, 12))
    colors = plt.get_cmap("rainbow")

    for (i, iou) in enumerate(r.mAPs):
        plt.plot(r.times, r.mAPs[iou], color=colors(float(i)/len(r.mAPs)), label="mAP@." + str(iou))

    plt.plot(r.times, r.AP, color='k', linewidth=2, label="AP")        

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0) 

    plt.xlabel('annotation time (minutes)')
    plt.ylabel('average precision')
  
    plt.legend()
    return fig, ax



if __name__ == '__main__':
    figure_path = "/home/oliver/sync/figures/running_maps"

    loaded = load_all(datasets, base_path)._without('aerial_penguins')
    
    summaries = pluck_struct('summary', loaded)
    results = compute_APs(loaded, sigma=5)

    for k, r in results.items():
        fig, ax = fig, ax = plot_APs_dataset(r)
        plt.title(k + ' - accuracy predictions vs. corrected images')
        fig.savefig(path.join(figure_path, k + ".pdf"), bbox_inches='tight')
      

    fig, ax = plot_running_mAPs(results, ious=[75])
    fig.savefig(path.join(figure_path, "overall.pdf"), bbox_inches='tight')

