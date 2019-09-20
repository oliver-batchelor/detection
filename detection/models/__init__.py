from tools import struct
from .retina.model import model as retina
from .ttf.model import model as ttf

def merge(*dicts):
    m = {}
    for d in dicts:
        m.update(d)

    return m

models = struct(retina=retina, ttf=ttf)
parameters = models._map(lambda m: m.parameters)
    
