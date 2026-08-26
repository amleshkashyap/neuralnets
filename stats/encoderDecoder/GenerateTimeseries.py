import numpy as np
from matplotlib import pyplot as plt
import torch

class GenerateTimeseries:
    def __init__(self):
        self.data = None
        self.XTrain = None
        self.XTest = None
        self.YTrain = None
        self.YTest = None

    def reset(self):
        self.data = None
        self.XTrain = None
        self.XTest = None
        self.YTrain = None
        self.YTest = None

    def getXTrain(self):
        return self.XTrain

    def getXTest(self):
        return self.XTest

    def getYTrain(self):
        return self.YTrain

    def getYTest(self):
        return self.YTest

    # Y(t) = sin(t) + (0.8 * cos(t/2)) + R(t) + 2.5
    def generateTimeseries(self, length):
        tf = 80 * np.pi
        t = np.linspace(0., tf, length)
        y = np.sin(t) + 0.8 * np.cos(0.5 * t) + np.random.normal(0., 0.3, length) + 2.5
        return y.tolist()

    def plot(self, length):
        data = self.generateTimeseries(length)
        plt.plot(data[:300])
        plt.savefig("timeseries.png")
        plt.show()

    def slidingWindow(self, ts, tsHistoryLength, tsTargetLength):
        X = []
        Y = []

        for i in range(tsHistoryLength + tsTargetLength, len(ts) + 1):
            X.append(ts[i - (tsHistoryLength + tsTargetLength): i - tsTargetLength])
            Y.append(ts[i - tsTargetLength: i])

        return X, Y

    def prepareData(self, tsLength, tsHistoryLength, tsTargetLength, testDsLength):
        self.reset()
        self.data = self.generateTimeseries(tsLength)
        X, Y = self.slidingWindow(self.data, tsHistoryLength, tsTargetLength)
        dsLength = len(X)
        def toTensor(ts):
            return torch.tensor(data = ts) \
                    .unsqueeze(2) \
                    .transpose(0, 1) \
                    .float()

        self.XTrain = toTensor(X[:dsLength - testDsLength])
        self.XTest = toTensor(X[dsLength - testDsLength:])
        self.YTrain = toTensor(Y[:dsLength - testDsLength])
        self.YTest = toTensor(Y[dsLength - testDsLength:])

if __name__ == "__main__":
    gt = GenerateTimeseries()
    gt.plot(2000)