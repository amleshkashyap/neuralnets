import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self,
                 inputSize,
                 hidden1Size,
                 hidden2Size,
                 conv1Size,
                 conv1KernelSize,
                 conv2KernelSize,
                 dropout,
                 outputSize):
        super(NeuralNet, self).__init__()
        self.conv1OutputChannel = conv1Size
        self.conv2OutputChannel = conv1Size * 2
        self.conv1Kernel = conv1KernelSize
        self.conv2Kernel = conv2KernelSize
        self.dropout = dropout
        self.pool = nn.MaxPool1d(kernel_size = 2)
        self.conv1 = nn.Conv1d(
            in_channels = 1,
            out_channels = self.conv1OutputChannel,
            kernel_size = self.conv1Kernel,
            padding = self.conv1Kernel - 1
        )
        self.conv2 = nn.Conv1d(
            in_channels = self.conv1OutputChannel,
            out_channels = self.conv2OutputChannel,
            kernel_size = self.conv2Kernel,
            padding = self.conv2Kernel - 1
        )
        # extract latent features from the prepared data using convolutional networks
        self.featureTensor = self.featureStack(torch.Tensor([[0] * inputSize]))
        # basic FFN/MLP
        self.linear1 = nn.Linear(self.featureTensor.size()[1], hidden1Size)
        self.linear2 = nn.Linear(hidden1Size, hidden2Size)
        self.linear3 = nn.Linear(hidden2Size, outputSize)

    def featureStack(self, x):
        x = x.unsqueeze(1)
        x = nn.functional.relu(self.pool(self.conv1(x)))
        x = nn.functional.relu(self.pool(self.conv2(x)))
        x = x.flatten(start_dim = 1)
        return x

    def fcStack(self, x):
        x = nn.functional.dropout(
            nn.functional.relu(self.linear1(x)),
            p = self.dropout
        )
        x = nn.functional.relu(self.linear2(x))
        return self.linear3(x).squeeze(-1)

    def forward(self,x):
        x = self.featureStack(x)
        return self.fcStack(x)