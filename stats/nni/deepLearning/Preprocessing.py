import os
import pandas as pd
import torch
import numpy as np

class Preprocessing:
    def __init__(self):
        self.data = None
        self.XTrain = None
        self.XValidate = None
        self.XTest = None
        self.YTrain = None
        self.YValidate = None
        self.YTest = None
        self.dirPath = os.path.dirname(os.path.realpath(__file__))

    def reset(self):
        self.data = None
        self.XTrain = None
        self.XValidate = None
        self.XTest = None
        self.YTrain = None
        self.YValidate = None
        self.YTest = None

    def getData(self):
        return self.data

    def getXTrain(self):
        return self.XTrain

    def getXValidate(self):
        return self.XValidate

    def getXTest(self):
        return self.XTest

    def getYTrain(self):
        return self.YTrain

    def getYValidate(self):
        return self.YValidate

    def getYTest(self):
        return self.YTest


    def slidingWindow(self, data, features):
        X = []
        Y = []

        for i in range(features + 1, len(data) + 1):
            X.append(data[i - (features + 1): i - 1])
            Y.append(data[i - 1])

        return X, Y

    def getPjmeTimeseries(self):
        df = pd.read_csv(f'{self.dirPath}/data/PJME_hourly.csv')
        ts = df['PJME_MW'].astype(int).values.reshape(-1, 1)[-1000:]
        return ts

    def prepareData(self, data, features, testLength):
        X, Y = self.slidingWindow(data, features)
        self.data = data

        self.XTrain, self.XTest, self.YTrain, self.YTest = \
            X[0:-testLength], X[-testLength:], Y[0:-testLength], Y[-testLength:]

        trainLength = round(len(self.XTrain) * 0.7)

        self.XTrain, self.XValidate, self.YTrain, self.YValidate = \
            self.XTrain[0:trainLength], self.XTrain[trainLength:], \
            self.YTrain[0:trainLength], self.YTrain[trainLength:]

        self.XTrain = torch.tensor(data = np.array(self.XTrain)).float()
        self.XValidate = torch.tensor(data = np.array(self.XValidate)).float()
        self.XTest = torch.tensor(data = np.array(self.XTest)).float()
        self.YTrain = torch.tensor(data = np.array(self.YTrain)).float()
        self.YValidate = torch.tensor(data = np.array(self.YValidate)).float()
        self.YTest = torch.tensor(data = np.array(self.YTest)).float()