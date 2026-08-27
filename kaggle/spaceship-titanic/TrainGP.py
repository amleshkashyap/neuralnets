import copy
import torch

class TrainGP:
    def __init__(self, model, likelihood, criterion):
        self.model = model
        self.likelihood = likelihood
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = 0.1
        )
        self.bestModel = None
        self.bestLikelihood = None

    def getBestModel(self):
        return self.bestModel

    def getBestLikelihood(self):
        return self.bestLikelihood

    def train(self, XTrain, YTrain, epochs):
        self.model.train()
        self.likelihood.train()
        minLoss = 1000
        minNoise = 1000
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(XTrain)
            # print(output.batch_shape) --> get shape
            loss = -self.criterion(output, YTrain).mean()
            loss.backward()
            noise = round(self.model.likelihood.noise.item(), 4)
            print(f'Iter: {epoch}/{epochs}, Loss: {round(loss.item(), 4)}, '
                  f'noise: {noise}')
            self.optimizer.step()
            if loss.item() < minLoss and noise < minNoise:
                self.bestModel = copy.deepcopy(self.model)
                self.bestLikelihood = copy.deepcopy(self.likelihood)
                minLoss = loss.item()
                minNoise = noise