import torch
import torch.nn as nn

class MLPerceptron2(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, hiddenLayers):
        super(MLPerceptron2, self).__init__()
        self.conv1OutputChannel = 6
        self.conv2OutputChannel = 12
        self.conv1Kernel = 4
        self.conv2Kernel = 2
        self.dropout = 0.1
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
        self.featureTensor = self.featureStack(torch.Tensor([[[0] * inputSize]]))
        self.hiddenLayer = nn.Linear(
            self.featureTensor.size()[1],
            hiddenSize
        )
        self.outputLayer = nn.Linear(
            hiddenSize,
            outputSize
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)

            # Optional: Ensure biases are set if they exist
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)

    def featureStack(self, x):
        # x = x.unsqueeze(1)
        x = nn.functional.relu(self.pool(self.conv1(x)))
        x = nn.functional.relu(self.pool(self.conv2(x)))
        x = x.flatten(start_dim = 1)
        return x

    def fcStack(self, x):
        x = nn.functional.dropout(
            nn.functional.relu(self.hiddenLayer(x)),
            p = self.dropout
        )
        return self.outputLayer(x)

    def forward(self,x):
        x = self.featureStack(x)
        return self.fcStack(x)