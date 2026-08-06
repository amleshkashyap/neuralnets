import os

import pandas as pd

from RecurrentNN import RecurrentNN
from GRU import GRU
from LSTM import LSTM
from MLPerceptron import MLPerceptron
from Train import Train
from Preprocess import Preprocess
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from Parameters import *
from CompareTrainTest import *

from Inference import Inference

if __name__ == "__main__":
    trainFilePath = ["data", "train.csv"]
    testFilePath = ["data", "test.csv"]

    # scaler to normalize the dataset
    scaler = StandardScaler()

    compareTrainTest = CompareTrainTest(trainFilePath, testFilePath, scaler)
    # compareTrainTest.loadData()
    # compareTrainTest.compareTrainTest()
    # os._exit(0)

    # preprocess the training dataset and the DataLoader
    preprocess = Preprocess(trainFilePath, scaler, {})
    preprocess.loadData()
    dataloader = preprocess.getData()
    categoryData = preprocess.getCategoryData()

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

    # os._exit(0)

    for col1 in trainDf.columns:
        for col2 in trainDf.columns:
            if col1 == col2:
                continue
            # print(col1, "\t", col2, "\t\t", round(trainDf[col1].corr(trainDf[col2]), 2))
            # correlation, pValue = stats.pointbiserialr(trainDf[col1], trainDf[col2])
            # if pValue < 0.05:
            #     print(col1, "\t", col2, "\t\t", round(correlation, 2))
            # else:
            #     print(col1, "\t", col2, "\t\t", "Irrelevant")

    # os._exit(0)

    # input features
    INPUT_SIZE = 19

    # binary classification has a single valued output
    OUTPUT_SIZE = 1

    # model = RecurrentNN(
    #     INPUT_SIZE,
    #     HIDDEN_SIZE,
    #     OUTPUT_SIZE,
    #     HIDDEN_LAYERS
    # )

    # model = GRU(
    #     INPUT_SIZE,
    #     HIDDEN_SIZE,
    #     OUTPUT_SIZE,
    #     HIDDEN_LAYERS
    # )

    # model = LSTM(
    #     INPUT_SIZE,
    #     HIDDEN_SIZE,
    #     OUTPUT_SIZE,
    #     HIDDEN_LAYERS
    # )

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

    trainer.train(EPOCHS)

    inference = Inference(model, criterion)
    inference.predict(preprocess.getXValidate(), preprocess.getYValidate())


    preprocessTest = Preprocess(testFilePath, preprocess.getScaler(), categoryData)
    preprocessTest.loadData('test')
    inference.predict(preprocessTest.getXTrain(), preprocessTest.getYTrain())