import random
import torch
import numpy as np
from numpy.distutils.fcompiler import none
import copy

random.seed(1)
np.random.seed(1)

class Preprocess:
    def __init__(self):
        self.data = None
        self.XTrain = None
        self.XValidate = None
        self.XTest = None
        self.YTrain = None
        self.YValidate = none
        self.YTest = None

    def reset(self):
        self.data = None
        self.XTrain = None
        self.XValidate = None
        self.XTest = None
        self.YTrain = None
        self.YValidate = none
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

    def generateTimeseries(self, length):
        # Y(t) = Y(t - 1) + R1(t - 1) + R1(t - 2) + 4 * R2(t - 3) * (R3(t - 4) + R3(t - 6))
        backshift = 10
        r1 = np.random.random(length + backshift)
        r2 = np.random.random(length + backshift)
        rm = [random.choices([0, 0, 0, 1])[0]
              for _ in range(length + backshift)]

        ts = np.zeros([length + backshift, 4])
        for i in range(backshift, length + backshift):
            ts[i - 1, 1] = r1[i - 1]
            ts[i - 1, 2] = r2[i - 1]
            ts[i - 1, 3] = rm[i - 1]

            ts[i, 0] = ts[i - 1, 0] - \
                       (r1[i - 1] + r1[i - 2]) + \
                       (4 * r2[i - 3]) * (rm[i - 4] + rm[i - 6])

        # generates shape - (5000, 4) - eg, [[1, 2, 3, 4], [4, 5, 6, 7], ...]
        return ts[backshift:]

    def tsDiff(self, ts):
        diffTs = [0] * len(ts)
        for i in range(1, len(ts)):
            diffTs[i] = ts[i] - ts[i - 1]
        return diffTs

    def tsInt(self, tsDiff, tsBase, start = 0):
        ts = []
        for i in range(len(tsDiff)):
            if i == 0:
                ts.append(start + tsDiff[0])
            else:
                ts.append(tsDiff[i] + tsBase[i - 1])
        return ts

    def getDifferencedData(self, data):
        self.reset()
        dataY = self.tsDiff(data[:, 0])
        self.data = copy.deepcopy(data)
        self.data[:, 0] = dataY
        return self.data

    def slidingWindow(self, ts, features, targetLength):
        X = []
        Y = []

        for i in range(features + targetLength, len(ts) + 1):
            X.append(ts[i - (features + targetLength): i - targetLength])
            Y.append(ts[i - targetLength: i])

        # converts dataset of (5000, 4) to (5000, 20, 4) - ie, picks 20 rows from original data
        #   and groups them together to form 1 datapoint of shape (1, 20, 4)
        #   finally, creates 5000 new points assuming data follows a timeseries
        return X, Y

    def prepareData(self, data, features, testLength, trainRatio = 0.7, targetLength = 1):
        self.reset()
        self.data = data
        X, Y = self.slidingWindow(self.data, features, targetLength)
        trainLength = round(len(self.data) * trainRatio)

        self.XTrain, self.YTrain, self.XTest, self.YTest = \
            X[0:-testLength], Y[0:-testLength], X[-testLength:], Y[-testLength:]

        self.XTrain, self.YTrain, self.XValidate, self.YValidate = \
            self.XTrain[0:trainLength], self.YTrain[0:trainLength], \
            self.XTrain[trainLength:], self.YTrain[trainLength:]

        # rotates the last two dimensions
        #   converts training data from shape (3500, 20, 4) to (3500, 4, 20)
        #   NOTE that the TCN layers accept - [[4, 10], [10, 10], [10, 10], [10, 10]] - hence,
        #   input is converted channel first
        self.XTrain = torch.tensor(data = np.array(self.XTrain)).float().transpose(1, 2)
        self.XValidate = torch.tensor(data = np.array(self.XValidate)).float().transpose(1, 2)
        self.XTest = torch.tensor(data = np.array(self.XTest)).float().transpose(1, 2)
        self.YTrain = torch.tensor(data = np.array(self.YTrain)).float()[:, :, 0]
        self.YValidate = torch.tensor(data = np.array(self.YValidate)).float()[:, :, 0]
        self.YTest = torch.tensor(data = np.array(self.YTest)).float()[:, :, 0]