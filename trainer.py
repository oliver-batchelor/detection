
import torch
from torch import nn

from tqdm import tqdm
import gc


def const(a):
    def f(*args):
        return a
    return f



            


def train(loader, eval, optimizer, hook = const(False)):
    results = []

    with tqdm() as bar:
        for n, data in enumerate(loader):
            optimizer.zero_grad()

            if hook(n, len(loader)): break

            result = eval(data)
            result.error.backward()
            optimizer.step()
            results.append(result.statistics)

            bar.update(result.size)
            if bar.total is None:
                bar.total = len(loader) * result.size

    return results


def update_bn(loader, eval):

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):

            eval(data)
            gc.collect()



def test(loader, eval, hook = const(False)):
    results = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):

            if hook(i, len(loader)): break

            result = eval(data)
            results.append(result)

            gc.collect()

        return results


def test_images(files, eval):
    results = []

    for (image_file, mask_file) in tqdm(files):
        data = dataset.load_rgb(image_file)
        label = dataset.load_label(mask_file)

        results.append((image_file, eval(data)))

    return results
