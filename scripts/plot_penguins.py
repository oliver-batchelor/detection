from scripts.load_figures import *
from scripts.plot_summaries import *
from scripts.plot_running_mAP import *
from scripts.plot_action_histograms import *
from scripts.plot_scatters import *

from scripts.datasets import match_datasets


if __name__ == '__main__':
    figure_path = "/home/oliver/sync/figures/aerial_penguins"


    loaded_a = load_all(penguins_a, base_path)
    loaded_b = load_all(penguins_b, base_path)

    combined = loaded_a._merge(loaded_b)

    pprint_struct(pluck_struct('summary', loaded_a))
    pprint_struct(pluck_struct('summary', loaded_b))

    match_threshold = 0.1
    matches = struct(
        royds = match_datasets(loaded_a.royds_a, loaded_b.royds_b, threshold=match_threshold),
        hallett = match_datasets(loaded_a.hallett_a, loaded_b.hallett_b, threshold=match_threshold),
        cotter = match_datasets(loaded_a.cotter_a, loaded_b.cotter_b, threshold=match_threshold)
    )

    [ print(k, m.total) for k, m in matches.items() ]
    

    fig, ax = actions_time_scatter(loaded_b, color_map=penguin_colors, labels=penguin_labels)
    ax.set_ylim(ymin=0, ymax=150)
    ax.set_xlim(xmin=0, xmax=50)
    fig.savefig(path.join(figure_path, "actions_time_b.pdf"), bbox_inches='tight')

    fig, ax = actions_time_scatter(loaded_a, color_map=penguin_colors, labels=penguin_labels)
    ax.set_ylim(ymin=0, ymax=150)
    ax.set_xlim(xmin=0, xmax=50)
    fig.savefig(path.join(figure_path, "actions_time_a.pdf"), bbox_inches='tight')

    fig, ax = plot_durations(combined, keys=penguin_keys, color_map=penguin_colors, labels=penguin_labels)
    fig.savefig(path.join(figure_path, "duration_boxplot.pdf"), bbox_inches='tight')


    ious=[50]
    results = compute_mAPs(combined, iou=ious, sigma=5)   

    fig, ax = plot_running_mAPs(results, ious=ious, color_map=penguin_colors, labels=penguin_labels)
    fig.savefig(path.join(figure_path, "running_mAP.pdf"), bbox_inches='tight')

    plot_action_histograms(combined, color_map=penguin_colors, labels=penguin_labels, figure_path=figure_path)
