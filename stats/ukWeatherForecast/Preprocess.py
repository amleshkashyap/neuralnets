import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Preprocess:
    def __init__(self):
        self.data = []
        self.XTrain = []
        self.XValidate = []
        self.XTest = []
        self.YTrain = []
        self.YValidate = []
        self.YTest = []

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

    def rawTimeSeries(self):
        dirPath = os.path.dirname(os.path.realpath(__file__))
        df = pd.read_csv(f'{dirPath}/data/MET_Office_Weather_Data.csv')
        self.data = df.loc[df['station'] == 'sheffield']['tmin'].tolist()

    def clearData(self):
        self.data = []

    def rawMetrics(self):
        self.clearData()
        self.rawTimeSeries()
        print(f'Count: {len(self.data)}')
        print(f'Min: {np.nanmin(self.data)}')
        print(f'Max: {np.nanmax(self.data)}')
        print(f'Mean: {round(np.nanmean(self.data), 2)}')
        print(f'Median: {round(np.nanmedian(self.data), 2)}')
        print(f'Stddev: {round(np.nanstd(self.data), 2)}')
        print(f'NA Values: {np.count_nonzero(np.isnan(self.data))}')

    def interpolatedData(self):
        dirPath = os.path.dirname(os.path.realpath(__file__))
        df = pd.read_csv(f'{dirPath}/data/MET_Office_Weather_Data.csv')
        self.data = df.loc[df['station'] == 'sheffield']['tmin'].interpolate().dropna().tolist()

    def plotInterpolatedData(self):
        self.clearData()
        self.interpolatedData()
        plt.plot(self.data[-120:])
        plt.savefig('interpolatedData.png')
        plt.show()

    def slidingWindow(self, windowLen):
        X = []
        Y = []
        for i in range(windowLen + 1, len(self.data) + 1):
            X.append(self.data[i - (windowLen + 1): i - 1])
            Y.append([self.data[i - 1]])
        return X, Y

    def prepareData(self, windowLen, testLength):
        self.clearData()
        self.interpolatedData()
        X, Y = self.slidingWindow(windowLen)
        self.XTrain, self.YTrain, self.XTest, self.YTest = \
            X[0:-testLength], Y[0:-testLength], X[-testLength:], Y[-testLength:]

        trainLength = round(len(self.data) * 0.7)
        self.XTrain, self.XValidate, self.YTrain, self.YValidate = \
            self.XTrain[0:trainLength], self.XTrain[trainLength:], self.YTrain[0:trainLength], self.YTrain[trainLength:]

        self.XTrain = torch.tensor(data = self.XTrain)
        self.XValidate = torch.tensor(data = self.XValidate)
        self.XTest = torch.tensor(data = self.XTest)
        self.YTrain = torch.tensor(data = self.YTrain).squeeze(-1)
        self.YValidate = torch.tensor(data = self.YValidate).squeeze(-1)
        self.YTest = torch.tensor(data = self.YTest).squeeze(-1)

if __name__ == '__main__':
    preprocess = Preprocess()
    preprocess.rawMetrics()
    preprocess.plotInterpolatedData()