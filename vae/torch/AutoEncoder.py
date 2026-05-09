import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # hidden units/latent dimensions
        self.num_hidden = 8

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),     # input sz: 784, output sz: 256
            nn.ReLU(),                                    # activation function
            nn.Linear(256, self.num_hidden),    # input sz: 256, output sz: 8
            nn.ReLU(),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.num_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),                                   # apply Sigmoid to compress output to range (0, 1)
        )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded