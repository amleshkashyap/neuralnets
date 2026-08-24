import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(LSTM, self).__init__()
        self.model = nn.LSTM(
            inputSize,
            hiddenSize,
            batch_first = True
        )
        self.fc = nn.Linear(
            hiddenSize,
            outputSize
        )

    def forward(self, x, h = None):
        out, h = self.model(x, h)
        lastHiddenStates = out[:, -1]
        out = self.fc(lastHiddenStates)
        return out.squeeze(-1), h