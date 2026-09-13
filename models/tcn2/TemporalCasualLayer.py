from typing import OrderedDict
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from models.Crop import Crop

class TemporalCasualLayer(nn.Module):
    def __init__(self,
                 nInputs,
                 nOutputs,
                 kernelSize,
                 stride,
                 dilation,
                 dropout = 0.2,
                 act = 'relu',
                 slices = 2,
                 useBias = True):
        super(TemporalCasualLayer, self).__init__()
        padding = (kernelSize - 1) * dilation
        convParams = {
            'kernel_size': kernelSize,
            'stride': stride,
            'padding': padding,
            'dilation': dilation
        }
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh()
        }

        self.useBias = useBias
        layers = OrderedDict()
        for s in range(1, slices + 1):
            if s == 1:
                layers[f'conv{s}'] = weight_norm(nn.Conv1d(nInputs, nOutputs, **convParams))
            else:
                layers[f'conv{s}'] = weight_norm(nn.Conv1d(nOutputs, nInputs, **convParams))

            layers[f'crop{s}'] = Crop(padding)
            layers[f'act{s}'] = activations[act]
            layers[f'dropout{s}'] = nn.Dropout(dropout)

        self.model = nn.Sequential(layers)

        if nInputs != nOutputs and useBias:
            self.bias = nn.Conv1d(nInputs, nOutputs, 1)
        else:
            self.bias = None

        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.model(x)
        if self.useBias:
            b = x if self.bias is None else self.bias(x)
            return self.relu(y + b)
        return self.relu(y)