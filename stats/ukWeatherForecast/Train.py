import copy
import sys
import matplotlib.pyplot as plt
import torch


class Train:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.modelPath = 'modelTmp.pt'
        self.trainingLoss = []
        self.validationLoss = []


    def train(self, XTrain, XValidate, YTrain, YValidate, epochs = 150):
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
            if epoch % 10 == 0:
                print(f'epoch {epoch}: train - {round(loss.item(), 4)}, validation - {round(valLoss.item(), 4)}')

    def makeCopy(self):
        torch.save(self.model, self.modelPath)

    def loadBestModel(self):
        return torch.load(
            self.modelPath,
            weights_only = False
        )

    def plotTrainingProgress(self):
        plt.title("Training Progress")
        plt.plot(self.trainingLoss, label = "Training Loss")
        plt.plot(self.validationLoss, label = "Validation Loss")
        plt.legend()
        plt.savefig("trainingProgress.png")
        plt.show()