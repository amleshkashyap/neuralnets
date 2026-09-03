import gpytorch
import pyro
from pyro.infer.mcmc import MCMC, NUTS

class BayesianGP(gpytorch.models.ExactGP):
    def __init__(self, XTrain, YTrain, likelihood):
        super(ExactGP, self).__init__(XTrain, YTrain, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(
            prior = gpytorch.priors.UniformPrior(-1, 1)
        )
        self.covar_module =  gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0),
                ard_num_dims = 17),
            outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15)
        )
        self.likelihood = likelihood

    def forward(self, x):
        meanX = self.mean_module(x)
        covarX = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(meanX, covarX)