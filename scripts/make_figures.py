import scripts.history
import scripts.datasets


def load_all(datasets, base_path):

    def load(filename):
        load_dataset(path.join(base_path, filename))


    loaded = datasets._map(load)

    

    return struct (
        summary = annotation_summary(loaded)
    



if __name__ == '__main__':

    base_path = '/home/oliver/export/'
    datasets = {
        #'penguins' : 'penguins.json',
        #'scallops' : 'scallops.json',
        hallett = 'penguins_hallett.json'
        cotter = 'penguins_cotter.json'
        royds = 'penguins_royds.json'
        combined = 'penguins_combined.json'
    }