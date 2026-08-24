import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(GRU, self).__init__()
        self.model = nn.GRU(
            inputSize,
            hiddenSize,
            batch_first = True
        )
        self.fc = nn.Linear(
            hiddenSize,
            outputSize
        )

    def forward(self, x, h = None):
        out, _ = self.model(x, h)
        lastHiddenStates = out[:, -1]
        out = self.fc(lastHiddenStates)
        return out.squeeze(-1), lastHiddenStates