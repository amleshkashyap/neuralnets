import gpytorch
from BayesianGP import BayesianGP
from sklearn.preprocessing import StandardScaler
from Preprocess import Preprocess

from InferenceGP import InferenceGP

import pyro
from pyro.infer.mcmc import MCMC, NUTS

# input features
INPUT_SIZE = 17

# binary classification has a single valued output
OUTPUT_SIZE = 1

def pyro_model(x, y):
    with gpytorch.settings.fast_computations(
            log_prob = False,
            covar_root_decomposition = False
    ):
        sampled_model = model.pyro_sample_from_prior()
        output = sampled_model(x)
        try:
            pyro.sample("obs", sampled_model.likelihood.pyro_marginal(x, output), obs = y)
        except Exception:
            # Catching numerical instability cases during random initialization steps
            pass
    return y

if __name__ == "__main__":
    trainFilePath = ["..", "data", "train.csv"]
    testFilePath = ["..", "data", "test.csv"]
    bestResPath = "../resultsBest.csv"

    scaler = StandardScaler()

    # preprocess the training dataset and the DataLoader
    preprocess = Preprocess(trainFilePath, scaler, {})
    preprocess.loadData()
    dataloader = preprocess.getData()
    categoryData = preprocess.getCategoryData()
    XTrain = preprocess.getXTrain().squeeze(1)
    YTrain = preprocess.getYTrain().squeeze(1)

    XValidate = preprocess.getXValidate().squeeze(1)
    YValidate = preprocess.getYValidate().squeeze(1)

    XTest = preprocess.getXTest().squeeze(1)
    YTest = preprocess.getYTest().squeeze(1)

    trainDf = preprocess.getDf()
    trainLabels = preprocess.getLabels().flatten().tolist()

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior = gpytorch.priors.GammaPrior(1.1, 0.5)
    )
    model = BayesianGP(
        XTrain,
        YTrain,
        likelihood
    )

    criterion = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    pyro.clear_param_store()
    nuts_kernel = NUTS(
        pyro_model,
        adapt_step_size = True
    )
    mcmc = MCMC(
        nuts_kernel,
        num_samples = 100,
        warmup_steps = 100,
        num_chains = 1
    )
    mcmc.run(XTrain, YTrain)

    # 5. Load the posterior samples back into the model
    # This transforms the single model into a batch GP model where each batch index is a sample
    model.pyro_load_from_samples(mcmc.get_samples())

    inference = InferenceGP(model, likelihood, criterion, bestResPath)

    inference.predict(XTest, YTest, 'validate')

    preprocessTest = Preprocess(testFilePath, preprocess.getScaler(), categoryData)
    preprocessTest.loadData('test')
    XTrain1 = preprocessTest.getXTrain().squeeze(1)
    YTrain1 = preprocessTest.getYTrain().squeeze(1)
    inference.predict(XTrain1, YTrain1)