import random
import matplotlib.pyplot as plt
from math import sin, cos
import numpy as np
from sklearn.linear_model import LinearRegression

class Processing:
    def __init__(self):
        pass

    def normalize(self, data):
        maxVal = max(data)
        minVal = min(data)
        normalised = [(v - minVal) / (maxVal - minVal) for v in data]
        return normalised, maxVal, minVal

    def denormalize(self, normalised, maxVal, minVal):
        denormalised = [v * (maxVal - minVal) + minVal for v in normalised]
        return denormalised

    def runNormalization(self):
        random.seed(1)
        data = [10 * sin(i) * cos(i) * cos(i) for i in range(20)]
        normalised, maxVal, minVal = self.normalize(data)
        denormalised = self.denormalize(normalised, maxVal, minVal)
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax1.set_title("Raw Time Series")
        ax1.plot(data)
        ax2.set_title("Normalized Time Series")
        ax2.plot(normalised)
        ax3.set_title("Denormalized Time Series")
        ax3.plot(denormalised)
        plt.show()

    def detrend(self, data):
        X = [[i] for i in range(len(data))]
        y = np.array(data).reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        a = reg.coef_[0][0]
        b = reg.intercept_[0]
        detrended = [(data[i] - a * i - b) for i in range(len(data))]
        return detrended, a, b

    def retrend(self, data, a, b):
        return [(data[i] + a * i - b) for i in range(len(data))]

    def runTrending(self):
        random.seed(1)
        data = [10 + 0.8 * i + sin(i) + 3 * random.random() for i in range(20)]
        detrended, a, b = self.detrend(data)
        retrended = self.retrend(detrended, a, b)
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax1.set_title("Raw Time Series")
        ax1.plot(data)
        ax2.set_title("Detrended Time Series")
        ax2.plot(detrended)
        ax3.set_title("Retrended Time Series")
        ax3.plot(retrended)
        plt.show()

    def differenced(self, data):
        differenced = [(data[i + 1] - data[i]) for i in range(len(data) - 1)]
        return differenced, data[0]

    def integrated(self, differenced, b):
        integrated = [b]
        for i in range(len(differenced)):
            integrated.append(differenced[i] + integrated[i])
        return integrated

    def runDifferencing(self):
        random.seed(1)
        data = [50 + 0.8 * i + 3 * sin(i) + 5 * random.random() for i in range(20)]
        differenced, b = self.differenced(data)
        integrated = self.integrated(differenced, b)
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax1.set_title("Raw Time Series")
        ax1.plot(data)
        ax2.set_title("Differenced Time Series")
        ax2.plot(differenced)
        ax3.set_title("Integrated Time Series")
        ax3.plot(integrated)
        plt.show()

if __name__ == "__main__":
    processing = Processing()
    processing.runNormalization()
    processing.runTrending()
    processing.runDifferencing()