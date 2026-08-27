import copy

import torch.optim
from statsmodels.tsa.base import prediction
import matplotlib.pyplot as plt


class Train:
    def __init__(self, model, criterion):
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = 0.005
        )
        self.criterion = criterion
        self.trainingLoss = []
        self.validationLoss = []
        self.bestModel = None

    def train(self, XTrain, XValidate, YTrain, YValidate, epochs):
        minValLoss = 1000000000
        self.model.train()
        for epoch in range(epochs):
            prediction = self.model(XTrain)
            loss = self.criterion(prediction, YTrain)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            valPrediction = self.model(XValidate)
            valLoss = self.criterion(valPrediction, YValidate)
            self.trainingLoss.append(loss.item())
            self.validationLoss.append(valLoss.item())
            if valLoss.item() < minValLoss:
                self.makeCopy()
                minValLoss = valLoss.item()
            if epoch % 100 == 0:
                # diff = (YTrain - prediction).view(-1).abs_().tolist()
                print(f'Epoch {epoch}: train - {round(loss.item(), 4)}, val loss - {round(valLoss.item(), 4)}')

    def makeCopy(self):
        self.bestModel = copy.deepcopy(self.model)

    def loadBestModel(self):
        return self.bestModel

    def plotTraining(self):
        plt.title('Training Progress')
        plt.yscale('log')
        plt.plot(self.trainingLoss, label = 'Training Loss')
        plt.plot(self.validationLoss, label = 'Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('trainingProgress.png')
        plt.show()