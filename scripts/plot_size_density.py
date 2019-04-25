from scripts.load_figures import *
from matplotlib.lines import Line2D

from scripts.datasets import *
from tools import to_structs

import scripts.figures

from dataset.imports.voc import import_voc
from dataset.imports.coco import import_coco

from scipy import stats


def plot_sizes_density(loaded, keys):
    fig, ax = make_chart()

    for k in keys:
        sizes = np.array(dataset_sizes(loaded[k])) * 100.0
        density = stats.kde.gaussian_kde(sizes,  bw_method=0.5)
   
        x = np.arange(0.0, 100, .4)
        plt.plot(x, density(x), label = k, color = dataset_colors[k])

    plt.xlabel('object size percent of image size')
    plt.ylabel('density')

    plt.title('distribution of object sizes in images')
    plt.legend()

    return fig, ax



if __name__ == '__main__':
    figure_path = "/home/oliver/sync/figures/summaries"
    
    loaded = load_all(datasets._subset("penguins", "branches", "scallops"), base_path)

    voc = to_structs(import_voc())
    coco = to_structs(import_coco(subsets = [('val2017', 'validate')]))    

    loaded = loaded._extend(pascal_voc=voc, coco=coco)
    keys = list(sorted(loaded.keys())) + ["coco", "pascal_voc"] 
  

    fig, ax = plot_sizes_density(loaded, keys)
    fig.savefig(path.join(figure_path, "sizes_density.pdf"), bbox_inches='tight')

