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
        # get the mean and variance for the latent distribution inferred by the encoder
        mu, sigma = self.encoder(x)

        # NOTE: get the standard deviation from variance
        std = torch.exp(0.5 * sigma)
        # NOTE: get the auxiliary random variable required to compute the latent vector
        epsilon = torch.randn_like(std)
        # NOTE: standard formula for getting z during VAE forward pass
        z = epsilon.mul(std).add_(mu)

        return self.decoder(z), mu, sigma