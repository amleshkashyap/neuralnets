import copy
import random
import torch

random.seed(1)
torch.manual_seed(1)

class Train:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.bestModel = copy.deepcopy(self.model)
        self.trainingLoss = []
        self.validationLoss = []

    def train(self, XTrain, XValidate, YTrain, YValidate, epochs = 1000):
        minValidationLoss = 100000000
        self.model.train()

        for epoch in range(epochs):
            prediction, _ = self.model(XTrain)
            loss = self.criterion(prediction, YTrain)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            validationPrediction, _ = self.model(XValidate)
            valLoss = self.criterion(validationPrediction, YValidate)
            self.trainingLoss.append(loss.item())
            self.validationLoss.append(valLoss.item())
            if valLoss.item() < minValidationLoss:
                self.bestModel = copy.deepcopy(self.model)
                minValidationLoss = valLoss.item()
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: training loss = {round(loss.item(), 4)}, validation loss = {round(valLoss.item(), 4)}')


    def getBestModel(self):
        return self.bestModel