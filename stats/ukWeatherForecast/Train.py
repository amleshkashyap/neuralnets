import copy
import sys


class Train:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.bestModel = copy.deepcopy(model)

    def train(self, XTrain, XValidate, YTrain, YValidate, epochs = 150):
        minValLoss = sys.maxsize
        trainingLoss = []
        validationLoss = []
        for epoch in range(epochs):
            prediction = self.model(XTrain)
            loss = self.criterion(prediction, YTrain)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            valPrediction = self.model(XValidate)
            valLoss = self.criterion(valPrediction, YValidate)
            trainingLoss.append(loss.item())
            validationLoss.append(valLoss.item())
            if valLoss.item() < minValLoss:
                self.bestModel = copy.deepcopy(self.model)
                minValLoss = valLoss.item()
            if epoch % 10 == 0:
                print(f'epoch {epoch}: train - {round(loss.item(), 4)}, validation - {round(valLoss.item(), 4)}')