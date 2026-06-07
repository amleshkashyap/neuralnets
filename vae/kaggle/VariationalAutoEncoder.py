import torch
import torch.nn as nn

from Encode import Encode
from Decode import Decode

class VariationalAutoEncoder(nn.Module):
    def __init__(self, channels, embeddings, kernel):
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = Encode(channels, embeddings, kernel)
        self.decoder = Decode(channels, embeddings, kernel)

    def forward(self, x):
        mu, sigma = self.encoder(x)

        # NOTE:
        std = torch.exp(0.5 * sigma)
        # NOTE:
        epsilon = torch.randn_like(std)
        # NOTE:
        z = epsilon.mul(std).add_(mu)

        return self.decoder(z), mu, sigma