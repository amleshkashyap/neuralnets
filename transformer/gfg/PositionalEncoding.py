import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, dModel, maxSeqLength):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxSeqLength, dModel)
        position = torch.arange(0, maxSeqLength, dtype = torch.float).unsqueeze(1)
        divTerm = torch.exp(torch.arange(0, dModel, 2).float() * -(math.log(10000.0) / dModel))
        pe[:, 0::2] = torch.sin(position * divTerm)
        pe[:, 1::2] = torch.cos(position * divTerm)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]