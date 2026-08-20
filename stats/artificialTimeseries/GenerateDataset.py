import torch
import random
from math import sin, cos
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

random.seed(1)
torch.manual_seed(1)

class GenerateDataset:
    def __init__(self):
        # governing equation
        #   T(t) = (0.2 * t + 300) + (5 * sin(t/5)) + (20 * cos(t/24)) + (100 * sin(t/120)) + (20 * R(t))
        #     - (0.2 * t + 300) - linear tend
        #     - (5 * sin(t/5)), (20 * cos(t/24)), (100 * sin(t/120)) - 3 seasonal periods
        #     - (20 * R(t)) - randomness
        self.data = []
        self.a = 0.2
        self.b = 300
        self.c = 5
        self.d = 20
        self.e = 100
        self.f = 20

        self.XTrain = []
        self.XValidate = []
        self.XTest = []
        self.YTrain = []
        self.YValidate = []
        self.YTest = []

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

    def generateData(self, count):
        random.seed(1)
        torch.manual_seed(1)
        for i in range(count):
            self.data.append(
                ((self.a * i) + self.b) +
                (self.c * sin(i/5)) +
                (self.d * cos(i/24)) +
                (self.e * sin(i/120)) +
                (self.f * random.random())
            )
        return self.data

    def clearData(self):
        self.data = []

    def plotData(self):
        self.clearData()
        self.generateData(3000)
        plt.plot(self.data)
        plt.savefig("timeseriesData.png")
        plt.show()

    def prepareData(self, windowLen, count):
        self.clearData()
        self.generateData(count)
        plt.plot(self.data)
        plt.savefig("timeseriesData.png")
        plt.show()
        X = []
        Y = []
        # use slices of the data (with length windowLen) as features, and the data value as output
        for i in range(windowLen + 1, count):
            X.append(self.data[i - (windowLen + 1) : i - 1])
            Y.append(self.data[i])

        self.XTrain, self.XTest, self.YTrain, self.YTest = train_test_split(
            X,
            Y,
            test_size = 0.3,
            shuffle = False
        )

        self.XValidate, self.XTest, self.YValidate, self.YTest = train_test_split(
            self.XTest,
            self.YTest,
            test_size = 0.5,
            shuffle = False
        )

        self.XTrain = torch.tensor(data = self.XTrain)
        self.XValidate = torch.tensor(data = self.XValidate)
        self.XTest = torch.tensor(data = self.XTest)
        self.YTrain = torch.tensor(data = self.YTrain)
        self.YValidate = torch.tensor(data = self.YValidate)
        self.YTest = torch.tensor(data = self.YTest)


if __name__ == "__main__":
    gd = GenerateDataset()
    gd.plotData()