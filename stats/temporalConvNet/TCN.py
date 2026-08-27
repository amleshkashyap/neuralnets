import torch.nn as nn
from TemporalConvolutionNetwork import TemporalConvolutionNetwork

class TCN(nn.Module):
    def __init__(self, inputSize, outputSize, channels, kernelSize, dropout):
        super(TCN, self).__init__()
        self.model = TemporalConvolutionNetwork(
            inputSize,
            channels,
            kernelSize,
            dropout
        )
        self.linear = nn.Linear(channels[-1], outputSize)

    def forward(self, x):
        y = self.model(x)
        return self.linear(y[:, :, -1])