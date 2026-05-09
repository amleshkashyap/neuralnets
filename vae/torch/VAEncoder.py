from AutoEncoder import AutoEncoder
import torch.nn as nn
import torch.nn.functional as F
import torch

class VAEncoder(AutoEncoder):
    def __init__(self):
        super().__init__()
        # add mean and log variance layers for reparameterization
        self.mean = nn.Linear(self.num_hidden, self.num_hidden)
        self.log_var = nn.Linear(self.num_hidden, self.num_hidden)

    def reparameterize(self, mu, log_var):
        # compute stddev from log_var
        stddev = torch.exp(0.5 * log_var)

        # generate random noise using the same shape as stddev
        eps = torch.randn_like(stddev)

        # reparameterized sample
        return mu + eps * stddev

    def forward(self, x):
        # encode
        encoded = self.encoder(x)

        # mean and log variance vectors
        mean = self.mean(encoded)
        log_var = self.log_var(encoded)

        # reparameterize latent variable
        z = self.reparameterize(mean, log_var)

        # pass latent variable via decoder
        decoded = self.decoder(z)

        return encoded, decoded, mean, log_var

    def sample(self, num_samples):
        with torch.no_grad():
            # generate random noise
            z = torch.randn(num_samples, self.num_hidden).to(self.device)

            # pass noise through decoder to generate samples
            samples = self.decoder(z)

        return samples

    # a loss function that combines binary cross-entropy and Kullback-Leibler divergence
    def loss_function(self, recon_x, x, mean, log_var):
        # binary cross entropy between reconstructed output and input - ie, reconstruction error
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        '''
        compute Kullback-Leibler divergence between learned latent variable distribution and std Gaussian distribution
          adds penalty for learned distribution to deviate too much from the prior distribution - prior distribution
          is a bunch of zero centered unit variance normal distributions
        '''

        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return BCE + KLD