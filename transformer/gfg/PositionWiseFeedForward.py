import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, dModel, dFf):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(dModel, dFf)
        self.fc2 = nn.Linear(dFf, dModel)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))