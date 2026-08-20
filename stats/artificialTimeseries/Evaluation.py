import matplotlib.pyplot as plt

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