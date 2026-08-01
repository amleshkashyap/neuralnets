import torch
import torch.nn as nn

class RecurrentNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.rnn = nn.GRU(
            input_size = inputSize,
            hidden_size = hiddenSize,
            batch_first = True,
        )
        self.fc = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        h0 = torch.zeros(
            1,
            x.size(0),
            self.hiddenSize
        )

        output, _ = self.rnn(x, h0)
        output = output[:, -1, :]
        output = self.fc(output)

        return output