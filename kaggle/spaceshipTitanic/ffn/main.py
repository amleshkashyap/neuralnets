from MLPerceptron import MLPerceptron
from Train import Train
from Preprocess import Preprocess
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from Parameters import *
from Inference import Inference


# input features
INPUT_SIZE = 17

# binary classification has a single valued output
OUTPUT_SIZE = 1

if __name__ == "__main__":
    trainFilePath = ["..", "data", "train.csv"]
    testFilePath = ["..", "data", "test.csv"]
    bestResPath = "../resultsBest.csv"

    # scaler to normalize the dataset
    scaler = StandardScaler()

    # preprocess the training dataset and the DataLoader
    preprocess = Preprocess(trainFilePath, scaler, {})
    preprocess.loadData()
    dataloader = preprocess.getData()
    categoryData = preprocess.getCategoryData()
    XValidate = preprocess.getXValidate()
    YValidate = preprocess.getYValidate()
    XTest = preprocess.getXTest()
    YTest = preprocess.getYTest()

    trainDf = preprocess.getDf()
    trainLabels = preprocess.getLabels().flatten().tolist()

    model = MLPerceptron(
        INPUT_SIZE,
        HIDDEN_SIZE,
        OUTPUT_SIZE,
        1
    )

    # binary cross entropy
    criterion = nn.BCEWithLogitsLoss()

    # simple adam optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr = LEARNING_RATE
    )

    trainer = Train(
        model,
        dataloader,
        criterion,
        optimizer
    )

    trainer.train(XValidate, YValidate, EPOCHS)
    bestModel = trainer.loadBestModel()

    inference = Inference(bestModel, criterion, bestResPath)
    inference.predict(preprocess.getXTest(), preprocess.getYTest(), 'validate')


    preprocessTest = Preprocess(testFilePath, preprocess.getScaler(), categoryData)
    preprocessTest.loadData('test')
    inference.predict(preprocessTest.getXTrain(), preprocessTest.getYTrain())