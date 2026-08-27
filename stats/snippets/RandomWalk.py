import matplotlib.pyplot as plt
import random

class RandomWalk:
    def __init__(self):
        pass

    def generateRandomWalk(self, length = 100, mu = 0, sigma = 1):
        ts = []
        for i in range(length):
            e = random.gauss(mu, sigma)
            if i == 0:
                ts.append(e)
            else:
                # R(t+1) = R(t) + E(t)
                ts.append(ts[i - 1] + e)

        return ts

if __name__ == "__main__":
    random.seed(10)
    rwalk = RandomWalk()
    walk = rwalk.generateRandomWalk()
    plt.plot(walk)
    plt.show()