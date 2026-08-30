import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from Crop import Crop

class TemporalCasualLayer(nn.Module):
    def __init__(self, inputSize, outputSize, kernelSize, stride, dilation, layerNum, dropout = 0.2):
        super(TemporalCasualLayer, self).__init__()
        padding = (kernelSize - 1) * dilation
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.dilation = dilation
        self.layerNum = layerNum
        convParams = {
            'kernel_size': kernelSize,
            'stride': stride,
            'padding': padding,
            'dilation': self.dilation
        }
        self.conv1 = weight_norm(nn.Conv1d(
            self.inputSize,
            self.outputSize,
            **convParams
        ))
        self.crop1 = Crop(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(
            self.outputSize,
            self.outputSize,
            **convParams
        ))
        self.crop2 = Crop(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.model = nn.Sequential(
            self.conv1,
            self.crop1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.crop2,
            self.relu2,
            self.dropout2
        )
        self.bias = nn.Conv1d(
            self.inputSize,
            self.outputSize,
            1
        ) if inputSize != outputSize else None
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.model(x)
        b = x if self.bias is None else self.bias(x)
        return self.relu(y + b)