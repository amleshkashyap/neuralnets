import torch
import torch.nn as nn
from Parameters import *

class RecurrentNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, hiddenLayers):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.rnn = nn.RNN(
            input_size = inputSize,
            hidden_size = self.hiddenSize,
            batch_first = True,
            num_layers = self.hiddenLayers,
        )
        self.fc = nn.Linear(self.hiddenSize, outputSize)

    def forward(self, x, h = None):
        # h0 = torch.zeros(
        #     self.hiddenLayers,
        #     x.size(0),
        #     self.hiddenSize
        # )

        output, _ = self.rnn(x, h)
        lastState = output[:, -1, :]
        output = self.fc(lastState)
        return output, lastState