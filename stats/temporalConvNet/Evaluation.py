import matplotlib.pyplot as plt
from Dummy import Dummy

class Evaluation:
    def __init__(self, model, criterion, preprocess, data, testLength):
        self.model = model
        self.criterion = criterion
        self.preprocess = preprocess
        self.data = data
        self.testLength = testLength

    def evaluate(self, XTest, YTest):
        self.model.eval()
        tcnPrediction = self.model(XTest)
        dummyPrediction = Dummy()(XTest)
        tcnLoss = round(self.criterion(tcnPrediction, YTest).item(), 4)
        dummyLoss = round(self.criterion(dummyPrediction, YTest).item(), 4)

        plt.title(f'Test| TCN: {tcnLoss}; Dummy: {dummyLoss}')
        plt.plot(
            self.preprocess.tsInt(
                tcnPrediction.view(-1).tolist(),
                self.data[-self.testLength:, 0],
                start = self.data[-self.testLength - 1, 0]
            ),
            label = 'TCN'
        )
        plt.plot(
            self.data[-self.testLength - 1:, 0],
            label = 'Real'
        )
        plt.legend()
        plt.savefig('evaluation.png')
        plt.show()