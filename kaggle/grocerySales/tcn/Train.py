import copy

import torch.optim
from statsmodels.tsa.base import prediction
import matplotlib.pyplot as plt


class Train:
    def __init__(self, model, criterion, modelPath):
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = 0.005
        )
        self.criterion = criterion
        self.trainingLoss = []
        self.validationLoss = []
        self.modelPath = modelPath

    def train(self, XTrain, XValidate, YTrain, YValidate, epochs, storeFamily):
        minValLoss = 1000000000
        stateDict = None
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
                stateDict = copy.deepcopy(self.model.state_dict())
                minValLoss = valLoss.item()
            if epoch % 20 == 0:
                print(f'Epoch {epoch}/{epochs} For {storeFamily}: train - {round(loss.item(), 4)}, val loss - {round(valLoss.item(), 4)}')
        self.makeCopy(stateDict)

    def makeCopy(self, stateDict):
        torch.save(
            stateDict,
            f = self.modelPath
        )

    def plotTraining(self):
        plt.title('Training Progress')
        plt.yscale('log')
        plt.plot(self.trainingLoss, label = 'Training Loss')
        plt.plot(self.validationLoss, label = 'Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        # plt.savefig('trainingProgress.png')
        # plt.show()