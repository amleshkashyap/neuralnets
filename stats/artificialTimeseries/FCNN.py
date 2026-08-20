import torch.nn as nn
import torch.nn.functional as functional
import torch
import random

random.seed(1)
torch.manual_seed(1)

class FCNN(nn.Module):
    def __init__(self, inputSize, hidden1Size, hidden2Size, outputSize):
        super(FCNN, self).__init__()
        self.linear1 = nn.Linear(inputSize, hidden1Size)
        self.linear2 = nn.Linear(hidden1Size, hidden2Size)
        self.linear3 = nn.Linear(hidden2Size, outputSize)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        return self.linear3(x).squeeze(-1)