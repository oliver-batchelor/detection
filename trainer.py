
import torch
from torch import nn

from tqdm import tqdm
import gc


def const(a):
    def f(*args):
        return a
    return f



def run_progress(loader, hook, eval):
    results = []

    with tqdm() as bar:
        for i, data in enumerate(loader):
            result = eval(data)

            if hook and hook((i + 1) * result.size, len(loader) * result.size): break

            results.append(result)
            bar.update(result.size)
            if bar.total is None:
                bar.total = len(loader) * result.size

    return results

def train(loader, eval, optimizer, hook = None):

    def update(data):
        optimizer.zero_grad()

        result = eval(data)
        result.error.backward()
        optimizer.step()

        return result.statistics
        
    return run_progress(loader, hook, update)


def update_bn(loader, eval):

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):

            eval(data)
            gc.collect()



def test(loader, eval, hook = None):

    with torch.no_grad():
        return run_progress(loader, hook, eval)

