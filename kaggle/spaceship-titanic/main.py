import os

import pandas as pd

from RecurrentNN import RecurrentNN
from GRU import GRU
from Train import Train
from Preprocess import Preprocess
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy import stats

from Inference import Inference

if __name__ == "__main__":
    trainFilePath = ["data", "train.csv"]
    testFilePath = ["data", "test.csv"]

    # scaler to normalize the dataset
    scaler = StandardScaler()

    # preprocess the training dataset and the DataLoader
    preprocess = Preprocess(trainFilePath, scaler)
    preprocess.loadData()
    dataloader = preprocess.getData()

    trainDf = preprocess.getDf()
    trainLabels = preprocess.getLabels().flatten().tolist()
    for col in trainDf.columns:
        correlation, pValue = stats.pointbiserialr(trainDf[col], pd.Series(trainLabels))
        if pValue < 0.05:
            print(col, "\t\t", round(correlation, 2))
        else:
            print(col, "\t\t", "Irrelevant")

    for col1 in trainDf.columns:
        for col2 in trainDf.columns:
            if col1 == col2:
                continue
            print(col1, "\t", col2, "\t\t", round(trainDf[col1].corr(trainDf[col2]), 2))
            # correlation, pValue = stats.pointbiserialr(trainDf[col1], trainDf[col2])
            # if pValue < 0.05:
            #     print(col1, "\t", col2, "\t\t", round(correlation, 2))
            # else:
            #     print(col1, "\t", col2, "\t\t", "Irrelevant")

    # os._exit(0)

    # input features
    INPUT_SIZE = 15

    # hidden layer features
    HIDDEN_SIZE = 20

    # binary classification has a single valued output
    OUTPUT_SIZE = 1

    # model = RecurrentNN(
    #     INPUT_SIZE,
    #     HIDDEN_SIZE,
    #     OUTPUT_SIZE
    # )
    model = GRU(
        INPUT_SIZE,
        HIDDEN_SIZE,
        OUTPUT_SIZE
    )

    # binary cross entropy
    criterion = nn.BCEWithLogitsLoss()

    # simple adam optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr = 0.005
    )

    trainer = Train(
        model,
        dataloader,
        criterion,
        optimizer
    )

    trainer.train(100)

    preprocessTest = Preprocess(testFilePath, preprocess.getScaler())
    preprocessTest.loadData()
    testDf = preprocessTest.getDf()

    inference = Inference(model, criterion)
    inference.predict(testDf, preprocessTest.getLabels())