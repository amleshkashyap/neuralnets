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
        self.dropout = nn.Dropout(
            p = 0.2
        )
        self.outputLayer = nn.Linear(
            hiddenSize,
            outputSize
        )
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavier Uniform initialization is non-zero and stable for training
            nn.init.kaiming_uniform_(module.weight)

            # Optional: Ensure biases are set if they exist
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)

    def forward(self, x):
        x = self.hiddenLayer(x)
        x = self.activation(x)
        # x = self.dropout(x)
        x = self.outputLayer(x)
        return x.squeeze(-1)