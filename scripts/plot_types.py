from scripts.make_figures import *
from matplotlib.lines import Line2D

import scripts.figures

import operator
from functools import reduce


figure_path = "/home/oliver/sync/figures/summaries"


if __name__ == '__main__':

    loaded = load_all(datasets._subset('penguins', 'branches'), base_path)
    keys = sorted(loaded.keys())

    pprint_struct(pluck_struct('summary', loaded))


    # fig, ax = instances_duration_scatter(loaded, ['apples1', 'apples2'])
    # ax.set_ylim(ymin=0)
    # ax.set_xlim(xmin=0)      
    # fig.savefig(path.join(figure_path, "instances_duration_scatter_apples.pdf"), bbox_inches='tight')


