import torch
from matplotlib import pyplot as plt

class Train:
    def __init__(self, model, name):
        self.name = name
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr = 0.02
        )
        self.criterion = torch.nn.MSELoss()
        self.trainingLoss = []
        self.validationLoss = []
        self.bestModelPath = f'{name}Tmp.pt'

    def makeCopy(self):
        torch.save(self.model, self.bestModelPath)

    def loadBestModel(self):
        return torch.load(
            self.bestModelPath,
            weights_only = False
        )

    def getCriterion(self):
        return self.criterion

    def getOptimizer(self):
        return self.optimizer

    def train(self, XTrain, XValidate, YTrain, YValidate, epochs = 100):
        self.model.train()
        minValLoss = 1000000000
        for epoch in range(epochs):
            prediction, _ = self.model(XTrain)
            loss = self.criterion(prediction, YTrain)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            valPrediction, _ = self.model(XValidate)
            valLoss = self.criterion(valPrediction, YValidate)
            self.trainingLoss.append(loss.item())
            self.validationLoss.append(valLoss.item())
            if valLoss.item() < minValLoss:
                self.makeCopy()
                minValLoss = valLoss.item()
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: training loss - {round(loss.item(), 4)}, validation loss - {round(valLoss.item(), 4)}')

    def plotTrainingProgress(self):
        plt.title('Training')
        plt.yscale('log')
        plt.plot(
            self.trainingLoss,
            label = 'Training Loss',
        )
        plt.plot(
            self.validationLoss,
            label = 'Validation Loss',
        )
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(f'train{self.name}.png')
        plt.show()