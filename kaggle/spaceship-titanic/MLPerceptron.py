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
        self.hidden2 = nn.Linear(
            hiddenSize,
            40
        )
        self.dropout = nn.Dropout(
            p = 0.1
        )
        self.outputLayer = nn.Linear(
            40,
            outputSize
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)

            # Optional: Ensure biases are set if they exist
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)

    def forward(self, x):
        x = self.hiddenLayer(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.outputLayer(x)
        return x.squeeze(-1)