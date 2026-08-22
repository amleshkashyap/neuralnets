import matplotlib.pyplot as plt
import torch.nn.functional as functional

class Evaluation:
    def __init__(self, fcnn, dummyPredictor, interpolationPredictor, hwesPredictor, criterion):
        self.fcnn = fcnn
        self.dummyPredictor = dummyPredictor
        self.interpolationPredictor = interpolationPredictor
        self.hwesPredictor = hwesPredictor
        self.criterion = criterion

    def compareModels(self, XTest, YTest):
        self.fcnn.eval()
        print("Evaluating All Models")
        print(f'FCNN Loss: {self.criterion(self.fcnn(XTest), YTest).item()}')
        print(f'Dummy Loss: {self.criterion(self.dummyPredictor(XTest), YTest).item()}')
        print(f'Interpolation Loss: {self.criterion(self.interpolationPredictor(XTest), YTest).item()}')
        print(f'HWES Loss: {self.criterion(self.hwesPredictor(XTest), YTest).item()}')

    def checkFCNNOnTraining(self, XTrain, YTrain):
        self.fcnn.eval()
        plt.title("FCNN On Training Dataset")
        plt.plot(YTrain, label = 'Actual')
        plt.plot(self.fcnn(XTrain).tolist(), label = 'FCNN Predicted')
        plt.legend()
        plt.savefig("evaluateOnTraining.png")
        plt.show()

    def evaluateOnTest(self, XTest, YTest):
        self.fcnn.eval()
        plt.title("FCNN vs HWES On Test Dataset")
        plt.plot(YTest, label = 'Actual')
        plt.plot(self.fcnn(XTest).tolist(), label = 'FCNN Predicted')
        plt.plot(self.hwesPredictor(XTest).tolist(), label = 'HWES Predicted')
        plt.legend()
        plt.savefig("evaluateOnTest.png")
        plt.show()

    def checkStdDev(self, XTest, YTest):
        length = len(YTest)
        fcnnAbsDev = (self.fcnn(XTest) - YTest).abs_()
        hwesAbsDev = (self.hwesPredictor(XTest) - YTest).abs_()
        diffPos = functional.relu(hwesAbsDev - fcnnAbsDev).reshape(length).tolist()
        diffMin = (-functional.relu(hwesAbsDev - fcnnAbsDev)).reshape(length).tolist()
        plt.title("FCNN vs HWES Predictor Comparison")
        plt.hlines(
            0,
            xmin = 0,
            xmax = length,
            linestyle = 'dashed',
        )
        plt.bar(
            list(range(length)),
            diffPos,
            color = 'g',
            label = 'FCNN Wins'
        )
        plt.bar(
            list(range(length)),
            diffMin,
            color = 'r',
            label = 'HWES Wins'
        )
        plt.legend()
        plt.savefig("checkStdDev.png")
        plt.show()