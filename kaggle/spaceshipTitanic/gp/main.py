import gpytorch
from ExactGP import ExactGP
from sklearn.preprocessing import StandardScaler
from Preprocess import Preprocess

from InferenceGP import InferenceGP
from TrainGP import TrainGP

# input features
INPUT_SIZE = 17

# binary classification has a single valued output
OUTPUT_SIZE = 1

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

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGP(
        XTrain,
        YTrain,
        likelihood
    )

    criterion = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    trainer = TrainGP(
        model,
        likelihood,
        criterion
    )

    trainer.train(XTrain, YTrain, 100)
    model = trainer.getBestModel()
    likelihood = trainer.getBestLikelihood()

    inference = InferenceGP(model, likelihood, criterion, bestResPath)

    inference.predict(XTest, YTest, 'validate')


    preprocessTest = Preprocess(testFilePath, preprocess.getScaler(), categoryData)
    preprocessTest.loadData('test')
    XTrain1 = preprocessTest.getXTrain().squeeze(1)
    YTrain1 = preprocessTest.getYTrain().squeeze(1)
    inference.predict(XTrain1, YTrain1)