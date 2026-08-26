import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, numLayers = 1):
        super(Decoder, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.numLayers = numLayers
        self.model = nn.LSTM(
            input_size = self.inputSize,
            hidden_size = self.hiddenSize,
            num_layers = self.numLayers
        )
        self.linear = nn.Linear(self.hiddenSize, self.outputSize)

    def forward(self, x, h):
        out, h = self.model(x.unsqueeze(0), h)
        y = self.linear(out.squeeze(0))
        return y, h