import torch.nn as nn
from TemporalCasualLayer import TemporalCasualLayer

class TemporalConvolutionNetwork(nn.Module):
    def __init__(self, inputSize, channels, kernelSize = 2, dropout = 0.2):
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        numLevels = len(channels)
        tclParams = {
            'kernelSize': kernelSize,
            'stride': 1,
            'dropout': dropout
        }
        for i in range(numLevels):
            dilation = 2 ** i
            inChannels = inputSize if i == 0 else channels[i - 1]
            outputChannels = channels[i]
            tclParams['dilation'] = dilation
            tclParams['layerNum'] = i
            tcl = TemporalCasualLayer(
                inChannels,
                outputChannels,
                **tclParams
            )
            layers.append(tcl)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)