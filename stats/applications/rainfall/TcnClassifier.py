import torch
import torch.nn as nn
from models.tcn2.TemporalConvolutionNetwork import TemporalConvolutionNetwork

class TcnClassifier(nn.Module):
    def __init__(self, **params):
        super(TcnClassifier, self).__init__()
        self.numChannels = params['numChannels']
        self.numClasses = params.pop('numClasses')
        self.model = TemporalConvolutionNetwork(**params)
        self.linear = nn.Linear(self.numChannels[-1], self.numClasses)

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x[:, :, -1])
        return torch.log_softmax(x, dim = 1)