from tools import struct
from .retina_net import model as retina_net

def merge(*dicts):
    m = {}
    for d in dicts:
        m.update(d)

    return m

models = struct(retina_net=retina_net)
parameters = models._map(lambda m: m.parameters)
    
