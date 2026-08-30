import gpytorch

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, XTrain, YTrain, likelihood):
        super(ExactGP, self).__init__(XTrain, YTrain, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims = 17)
        )
        self.likelihood = likelihood

    def forward(self, x):
        meanX = self.mean_module(x)
        covarX = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(meanX, covarX)