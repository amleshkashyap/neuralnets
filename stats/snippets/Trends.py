import matplotlib.pyplot as plt
import random
from math import log, sin

class Trends:
    def __init__(self):
        pass

    def linearTrend(self):
        random.seed(10)
        length = 50
        A = 5
        B = 0.5
        C = 3
        trend = [A + B * i for i in range(length)]
        noise = [C * random.gauss(0, 1) for _ in range(length)]
        ts = [trend[i] + noise[i] for i in range(length)]
        plt.plot(ts)
        plt.plot(trend)
        plt.show()

    def nonLinearTrend(self):
        random.seed(10)
        length = 100
        A = 2
        B = 25
        C = 5
        trend = [A + B * log(i) for i in range(1, length + 1)]
        noise = [C * random.gauss(0, 1) for _ in range(length)]
        ts = [trend[i] + noise[i] for i in range(length)]
        plt.plot(ts)
        plt.plot(trend)
        plt.show()

    def seasonalTrend(self):
        random.seed(10)
        length = 100
        A = 50
        B = -0.05
        C = 1
        S = 3
        trend = [A + B * i for i in range(length)]
        noise = [C * random.gauss(0, 1) for _ in range(length)]
        seasons = [S * sin(i / 5) for i in range(length)]
        ts = [trend[i] + noise[i] + seasons[i] for i in range(length)]
        plt.plot(ts)
        plt.plot(trend)
        plt.show()

if __name__ == "__main__":
    trends = Trends()
    trends.linearTrend()
    trends.nonLinearTrend()
    trends.seasonalTrend()