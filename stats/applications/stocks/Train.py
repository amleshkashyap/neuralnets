import torch.nn as nn
import torch
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

    def train(self,
              closingTrain,
              closingValidate,
              indicatorTrain,
              indicatorValidate,
              yTrain,
              yValidate,
              epochs
        ):
        minValLoss = 1000000000
        self.model.train()
        for epoch in range(epochs):
            predicted = self.model(closingTrain, indicatorTrain)
            loss = self.criterion(predicted, yTrain)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            valPredicted = self.model(closingValidate, indicatorValidate)
            valLoss = self.criterion(valPredicted, yValidate)
            self.trainingLoss.append(loss.item())
            self.validationLoss.append(valLoss.item())
            if valLoss.item() < minValLoss:
                minValLoss = valLoss.item()
                self.makeCopy()
            if epoch % 10 == 0:
                print(f'Epoch {epoch}| train: {round(loss.item(), 4)} | val: {round(valLoss.item(), 4)}')

    def plotTraining(self):
        plt.title('Training Progress')
        plt.plot(self.trainingLoss, label = 'Training Loss')
        plt.plot(self.validationLoss, label = 'Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('trainingProgress.png')
        plt.show()