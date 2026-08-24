import os
import torch
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

class Preprocess:
    def __init__(self):
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = os.path.join(self.dirPath, "data")
        self.data = []
        self.XTrain = None
        self.XValidate = None
        self.XTest = None
        self.YTrain = None
        self.YValidate = None
        self.YTest = None

    def reset(self):
        self.data = []
        self.XTrain = None
        self.XValidate = None
        self.XTest = None
        self.YTrain = None
        self.YValidate = None
        self.YTest = None

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

    def slidingWindow(self, data, windowLen):
        X = []
        Y = []

        for i in range(windowLen + 1, len(data) + 1):
            X.append(data[i - (windowLen + 1): i - 1])
            Y.append(data[i - 1])

        return X, Y

    def getPjmeTimeseries(self):
        df = pd.read_csv(os.path.join(self.dirPath, "PJME_hourly.csv"))
        data = df['PJME_MW'].astype(int).values.reshape(-1, 1)[-3000:]
        return data

    def getAepTimeseries(self):
        df = pd.read_csv(os.path.join(self.dirPath, "AEP_hourly.csv"))
        data = df['AEP_MW'].astype(int).values.reshape(-1, 1)[-3000:]
        return data

    def getNiTimeseries(self):
        df = pd.read_csv(os.path.join(self.dirPath, "NI_hourly.csv"))
        data = df['NI_MW'].astype(int).values.reshape(-1, 1)[-3000:]
        return data

    def prepareTrainingData(self, data, windowLen, testLength):
        self.reset()
        X, Y = self.slidingWindow(data, windowLen)

        self.XTrain, self.XTest, self.YTrain, self.YTest = \
                X[0:-testLength], X[testLength:], Y[0:-testLength], Y[testLength:]

        trainLength = round(len(data) * 0.7)

        self.XTrain, self.XValidate, self.YTrain, self.YValidate = \
                self.XTrain[0:trainLength], self.XTrain[trainLength:], self.YTrain[0:trainLength], self.YTrain[trainLength:]

        self.XTrain = torch.tensor(data = np.array(self.XTrain)).float()
        self.XValidate = torch.tensor(data = np.array(self.XValidate)).float()
        self.XTest = torch.tensor(data = np.array(self.XTest)).float()
        self.YTrain = torch.tensor(data = np.array(self.YTrain)).float().squeeze(1)
        self.YValidate = torch.tensor(data = np.array(self.YValidate)).float().squeeze(1)
        self.YTest = torch.tensor(data = np.array(self.YTest)).float().squeeze(1)

    def viewAepTimeseries(self):
        data = self.getAepTimeseries()
        plt.title('AEP Hourly')
        plt.plot(data[:500])
        plt.savefig('AEP_hourly.png')
        plt.show()

    def viewPjmeTimeseries(self):
        data = self.getPjmeTimeseries()
        plt.title('PJME Hourly')
        plt.plot(data[:500])
        plt.savefig('PJME_hourly.png')
        plt.show()

    def viewNiTimeseries(self):
        data = self.getNiTimeseries()
        plt.title('NI Hourly')
        plt.plot(data[:500])
        plt.savefig('NI_hourly.png')
        plt.show()

if __name__ == '__main__':
    preprocess = Preprocess()
    preprocess.viewAepTimeseries()
    preprocess.viewPjmeTimeseries()
    preprocess.viewNiTimeseries()