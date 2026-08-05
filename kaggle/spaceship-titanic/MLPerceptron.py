import torch
import torch.nn as nn

class MLPerceptron(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, hiddenLayers):
        super().__init__()
        self.hiddenLayer = nn.Linear(
            inputSize,
            hiddenSize
        )
        self.activation = nn.ReLU()
        self.outputLayer = nn.Linear(
            hiddenSize,
            outputSize
        )

    def forward(self, x):
        x = self.hiddenLayer(x)
        x = self.activation(x)
        x = self.outputLayer(x)
        return x.squeeze(-1)