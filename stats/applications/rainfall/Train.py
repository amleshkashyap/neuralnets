import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt

class Train:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.bestModelPath = 'model.pt'
        self.trainingLoss = []
        self.validationLoss = []

    def makeCopy(self):
        torch.save(
            self.model.state_dict(),
            self.bestModelPath
        )

    def train(self, XTrain, XValidate, YTrain, YValidate, epochs):
        minValLoss = 1000000000
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
                print(f'Epoch: {epoch}| test: {round(loss.item(), 4)}| val: {round(valLoss.item(), 4)}')

        return minValLoss

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