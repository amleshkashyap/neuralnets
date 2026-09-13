import torch.nn as nn
from models.tcn2.TemporalCasualLayer import TemporalCasualLayer

class TemporalConvolutionNetwork(nn.Module):
    def __init__(self,
                 numInputs,
                 numChannels,
                 kernelSize = 2,
                 dropout = 0.2,
                 slices = 2,
                 act = 'relu',
                 useBias = True):
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        numLevels = len(numChannels)

        tclParams = {
            'kernelSize': kernelSize,
            'stride': 1,
            'dropout': dropout,
            'slices': slices,
            'act': act,
            'useBias': useBias
        }

        for i in range(numLevels):
            dilation = 2**i
            inChannels = numInputs if i == 0 else numChannels[i - 1]
            outChannels = numChannels[i]
            tclParams['dilation'] = dilation
            tcl = TemporalCasualLayer(inChannels, outChannels, **tclParams)
            layers.append(tcl)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)