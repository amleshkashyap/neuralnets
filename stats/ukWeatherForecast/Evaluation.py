from matplotlib import pyplot as plt

class Evaluation:
    def __init__(self, neuralNet, sarima, hwes, criterion):
        self.neuralNet = neuralNet
        self.sarima = sarima
        self.hwes = hwes
        self.criterion = criterion
        self.netPrediction = None
        self.sarimaPrediction = None
        self.hwesPrediction = None
        self.netAbsLoss = None
        self.sarimaAbsLoss = None
        self.hwesAbsLoss = None

    def evaluateOnTest(self, XTest, YTest):
        self.neuralNet.eval()
        self.netPrediction = self.neuralNet(XTest)
        self.sarimaPrediction = self.sarima(XTest)
        self.hwesPrediction = self.hwes(XTest)
        self.netAbsLoss = round(self.criterion(self.netPrediction, YTest).item(), 4)
        self.sarimaAbsLoss = round(self.criterion(self.sarimaPrediction, YTest).item(), 4)
        self.hwesAbsLoss = round(self.criterion(self.hwesPrediction, YTest).item(), 4)
        print('=======')
        print('Results')
        print(f'Neural Net Loss: {self.netAbsLoss}')
        print(f'SARIMAX Loss: {self.sarimaAbsLoss}')
        print(f'HWES Loss: {self.hwesAbsLoss}')

    def checkAgainstActual(self, YTest):
        self.neuralNet.eval()
        plt.title('Predicted Vs Actual On Test Dataset')
        plt.plot(
            YTest,
            '--',
            label = 'Actual',
            linewidth = 3
        )
        plt.plot(
            self.netPrediction.tolist(),
            label = 'Neural Net',
            color = 'g'
        )
        plt.plot(
            self.sarimaPrediction.tolist(),
            label = 'SARIMAX',
            color = 'r'
        )
        plt.plot(
            self.hwesPrediction.tolist(),
            label = 'HWES',
            color = 'brown'
        )
        plt.legend()
        plt.savefig('compareActual.png')
        plt.show()

    def checkStdDev(self, YTest):
        self.neuralNet.eval()
        length = len(YTest)
        netAbsDev = (self.netPrediction - YTest).abs_()
        sarimaDev = (self.sarimaPrediction - YTest).abs_()
        hwesDev = (self.hwesPrediction - YTest).abs_()
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax1.set_title(f'Neural Net: {self.netAbsLoss}')
        ax1.bar(
            list(range(length)),
            netAbsDev.view(length).tolist(),
            color = 'g'
        )
        ax2.set_title(f'SARIMAX: {self.sarimaAbsLoss}')
        ax2.bar(
            list(range(length)),
            sarimaDev.view(length).tolist(),
            color = 'r'
        )
        ax3.set_title(f'HWES: {self.hwesAbsLoss}')
        ax3.bar(
            list(range(length)),
            hwesDev.view(length).tolist(),
            color = 'brown'
        )
        plt.savefig('compareStdDev.png')
        plt.show()