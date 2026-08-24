import torch
import numpy as np
from matplotlib import pyplot as plt

class Evaluation:
    def __init__(self, model, criterion, name, scaler):
        self.model = model
        self.criterion = criterion
        self.scaler = scaler
        self.predicted = []
        self.name = name

    def evaluate(self, XValidate, XTest):
        self.model.eval()
        _, hList = self.model(XValidate)
        h = (hList[-1, :]).unsqueeze(-2)

        for seq in XTest.tolist():
            x = torch.tensor(data = [seq])
            y, h = self.model(x, h.unsqueeze(-2))
            unscaled = self.scaler.inverse_transform(np.array(y.item()).reshape(-1, 1))[0][0]
            self.predicted.append(unscaled)

    def plotEvaluation(self, YTest):
        real = self.scaler.inverse_transform(np.array(YTest).reshape(-1, 1))
        plt.title("Test Dataset")
        plt.plot(real, label = 'Actual')
        plt.plot(self.predicted, label = 'Predicted')
        plt.legend()
        plt.savefig(f'evaluation{self.name}.png')
        plt.show()