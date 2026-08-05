import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, hiddenLayers):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.lstm = nn.LSTM(
            input_size = inputSize,
            hidden_size = self.hiddenSize,
            batch_first = True,
            num_layers = self.hiddenLayers,
        )
        self.fc = nn.Linear(self.hiddenSize, outputSize)

    def forward(self, x):
        h0 = torch.zeros(
            self.hiddenLayers,
            x.size(0),
            self.hiddenSize
        )

        c0 = torch.zeros(
            self.hiddenLayers,
            x.size(0),
            self.hiddenSize
        )

        output, (hn, cn) = self.lstm(x, (h0, c0))
        output = output[:, -1, :]
        output = self.fc(output)

        return output