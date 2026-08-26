import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers = 1):
        super(Encoder, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.model = nn.LSTM(
            input_size = self.inputSize,
            hidden_size = self.hiddenSize,
            num_layers = self.numLayers
        )

    def forward(self, x):
        flat = x.view(x.shape[0], x.shape[1], self.inputSize)
        out, h = self.model(flat)
        return out, h