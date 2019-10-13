import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from functools import partial
from tools.model import match_size_2d,  centre_crop

from tools import Struct, shape

def identity(x, **kwargs):
    return x

def reverse(xs):
    return list(reversed(xs))

def image_size(inputs):
    if torch.is_tensor(inputs):
        assert inputs.dim() in [3, 4]

        if inputs.dim() == 3:
            return inputs.size(1), inputs.size(0)
        else:
            return inputs.size(2), inputs.size(1)
        

    assert (len(inputs) == 2)
    return inputs

def map_modules(m, type, f):
    if isinstance(m, type):
        return f(m)

    for k, v in m._modules.items():
        m._modules[k] = map_modules(m._modules[k], type, f)

    return m

def replace_batchnorms(m, num_groups):
    def convert(b):
        g = nn.GroupNorm(num_groups, b.num_features)
        g.weight = b.weight
        g.bias = b.bias

        return g

    return map_modules(m, nn.BatchNorm2d, convert)


class Lift(nn.Module):
    def __init__(self, f, **kwargs):
        super().__init__()

        self.kwargs = kwargs
        self.f = f

    def forward(self, input):
        return self.f(input, **self.kwargs)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def cascade(modules, input):
    outputs = []

    for module in modules:
        input = module(input)
        outputs.append(input)

    return outputs

class Cascade(nn.Sequential):
    def __init__(self, *args, drop_initial=0):
        super(Cascade, self).__init__(*args)
        self.drop = drop_initial


    def forward(self, input):
        out = cascade(self._modules.values(), input)
        return out[self.drop:]


class UpCascade(nn.Module):
    def __init__(self, *decoders):
        super(UpCascade, self).__init__()

        self.decoders = nn.Sequential(*decoders)

    def forward(self, inputs):

        assert len(inputs) == len(self.decoders)

        input = None
        outputs = []

        for module, skip in zip(reverse(self.decoders._modules.values()), reverse(inputs)):
            input = module(input, skip)
            outputs.append(input)

        return reverse(outputs)


class Parallel(nn.Module):
    def __init__(self, *modules):
        super(Parallel, self).__init__()
        self.parallel = nn.Sequential(*modules)

    def forward(self, inputs):
        assert len(inputs) == len(self.parallel)
        assert type(inputs) is list, "type of inputs is: " + str(type(inputs))
        
        return [m(i) for m, i in zip(self.parallel, inputs)]


class Named(nn.Module):
    def __init__(self, **named):
        super(Named, self).__init__()
        
        for k, v in named.items():
            self.add_module(k, v)

    def forward(self, input):
        output = {k: module(input) for k, module in self.named_children()}
        return Struct(output)


class Shared(nn.Module):
    def __init__(self, module):
        super(Shared, self).__init__()
        self.module = module

    def forward(self, inputs):
        return [self.module(input) for  input in inputs]


class Residual(nn.Sequential):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input):
        output = self.module(input)
        # assert (output.size(1) == input.size(1))

        return output + input


class Lookup(nn.Module):
    def __init__(self, k):
        super().__init__() 
        self.k = k

    def forward(self, inputs):
        return inputs[self.k]


class Conv(nn.Module):
    def __init__(self, in_size, out_size, kernel=3, stride=1, padding=None, bias=False, activation=nn.ReLU(inplace=True), groups=1):
        super().__init__()

        padding = kernel//2 if padding is None else padding

        self.norm = nn.BatchNorm2d(in_size)
        self.conv = nn.Conv2d(in_size, out_size, kernel, stride=stride, padding=padding, bias=bias, groups=1)
        self.activation = activation

    def forward(self, inputs):
        return self.conv(self.activation(self.norm(inputs)))


class LocalSE(nn.Module):

    def __init__(self, features, kernel = 7):
        super().__init__()

        self.conv1 = nn.Conv2d(features, features, 1)
        self.conv2 = nn.Conv2d(features, features, 1)
        self.kernel = kernel
        self.norm = nn.BatchNorm2d(features)


    def forward(self, inputs):
        x = F.avg_pool2d(inputs, self.kernel, stride=1, padding=self.kernel//2)

        x = self.norm(self.conv1(x))
        x = F.relu(x, inplace=True)
        x = torch.sigmoid(self.conv2(x))

        return inputs * x


class GlobalSE(nn.Module):

    def __init__(self, features):
        super().__init__()

        self.conv1 = Conv(features, features, 1)
        self.conv2 = Conv(features, features, 1)


    def forward(self, inputs):
        avg = F.adaptive_avg_pool2d(inputs, (1, 1)) 

        x = self.conv1(avg)
        x = torch.sigmoid(self.conv2(x))

        return inputs * x


def dropout(p=0.0):
    return nn.Dropout2d(p=p) if p > 0 else Lift(identity)


def basic_block(in_size, out_size):
    return nn.Sequential(
        Conv(in_size, out_size, activation=identity), 
        Conv(out_size, out_size), 
        nn.BatchNorm2d(out_size)
    )
    

def se_block(in_size, out_size):
    return nn.Sequential(
        basic_block(in_size, out_size),
        GlobalSE(out_size)
    )


def reduce_features(in_size, out_size, steps=2, kernel=1):
    def interp(i):
        t = i / steps
        d = out_size + int((1 - t) * (in_size - out_size))
        return d

    m = nn.Sequential(*[Conv(interp(i), interp(i + 1), kernel) for i in range(0, steps)])
    return m


def unbalanced_add(x, y):
    if x.size(1) > y.size(1):
        x = x.narrow(0, 1, y.size(1))
    elif y.size(1) < x.size(1):
        y = y.narrow(0, 1, x.size(1))

    return x + y


class Bias2d(nn.Module):
    def __init__(self, features, inplace=False):
        self.bias = Parameter(torch.Tensor(out_channels))
        self.inplace = inplace

    def forward(input):

        if self.inplace:
            return input.add_(self.bias)
        else:
            return input + self.bias



class Upscale(nn.Module):
    def __init__(self, features, scale_factor=2):
        super().__init__()
        self.conv = Conv(features, features * scale_factor**2)
        self.scale_factor = scale_factor

    def forward(self, inputs):
        return F.pixel_shuffle(self.conv(inputs), self.scale_factor)


class DecodeAdd(nn.Module):
    def __init__(self, features, module=None, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.module = module or identity
        #self.upscale = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.upscale = Upscale(features, scale_factor=scale_factor)

    def forward(self, inputs, skip):
        if not (inputs is None):
            upscaled = self.upscale(inputs)
            upscaled = match_size_2d(upscaled, skip)
            return self.module(skip + upscaled)

        return self.module(skip)


class Decode(nn.Module):
    def __init__(self, features, module=None, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.reduce = Conv(features * 2, features)
        self.module = module or identity
        #self.upscale = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.upscale = Upscale(features, scale_factor=scale_factor)


    def forward(self, inputs, skip):
        if not (inputs is None):
            #upscaled = F.upsample(inputs, scale_factor=self.scale_factor)
            upscaled = self.upscale(inputs)
            upscaled = match_size_2d(upscaled, skip)

            return self.module(self.reduce(torch.cat([upscaled, skip], 1)))

        return self.module(skip)



def init_weights(module):
    def f(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            init.kaiming_normal(m.weight)
    module.apply(f)
