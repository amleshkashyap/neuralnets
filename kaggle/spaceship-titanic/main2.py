import gpytorch
from ExactGP import ExactGP
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
from CompareTrainTest import *
from Preprocess import Preprocess

from InferenceGP import InferenceGP
from TrainGP import TrainGP

# input features
INPUT_SIZE = 17

# binary classification has a single valued output
OUTPUT_SIZE = 1

if __name__ == "__main__":
    trainFilePath = ["data", "train.csv"]
    testFilePath = ["data", "test.csv"]
    bestResPath = "resultsBest.csv"

    scaler = StandardScaler()

    compareTrainTest = CompareTrainTest(trainFilePath, testFilePath, scaler)

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
    idx = 0
    for col in trainDf.columns:
        correlation, pValue = stats.pointbiserialr(trainDf[col], pd.Series(trainLabels))
        if pValue < 0.05:
            print(col, "\t\t", round(correlation, 2))
        else:
            print(col, "\t\t", "Irrelevant")
        dummyMatrix = np.zeros((len(trainDf), len(trainDf.columns)))
        dummyMatrix[:, idx] = trainDf[col].values
        transformedMatrix = scaler.transform(dummyMatrix)
        toPlot = transformedMatrix[:, idx]
        plt.plot(toPlot)
        plt.xlabel("Datapoint")
        plt.ylabel(col)
        plt.grid(True)
        # plt.show()
        idx += 1

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