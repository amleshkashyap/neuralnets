import matplotlib.pyplot as plt
import seaborn as sns

class Visualize:
    def __init__(self):
        pass

    @staticmethod
    def plotBasicHistogram(data, bins):
        sns.histplot(
            data,
            bins = bins,
            kde = True,
            color = 'skyblue',
            edgecolor = 'black'
        )
        plt.xlabel('Values')
        plt.ylabel('Density')
        plt.show()