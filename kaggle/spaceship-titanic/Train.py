from Parameters import *
import torch

class Train:
    def __init__(self, model, dataloader, criterion, optimizer):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainingLoss = []
        self.validationLoss = []
        self.modelPath = 'modelTmp.pt'

    def train(self, XValidate, YValidate, epochs = EPOCHS):
        minValidationLoss = 1000000000
        for epoch in range(epochs):
            epochLoss = 0.0
            for batchX, batchY in self.dataloader:
                # run the forward method to do the prediction
                y = self.model(batchX)

                # compute the loss values
                loss = self.criterion(y, batchY)

                # Start BackPropagation
                # reset the gradients
                self.optimizer.zero_grad()
                # compute the gradients
                loss.backward()
                # update the weights using the above gradients and optimizer
                self.optimizer.step()
                epochLoss += loss.item() * batchX.size(0)
            valPrediction = self.model(XValidate)
            valLoss = self.criterion(valPrediction, YValidate)
            totalEpochLoss = epochLoss / len(self.dataloader.dataset)
            self.trainingLoss.append(totalEpochLoss)
            self.validationLoss.append(valLoss)
            if valLoss < minValidationLoss:
                self.makeCopy()
                minValidationLoss = valLoss.item()

            print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {totalEpochLoss:.4f} | ValLoss: {valLoss.item():.4f}")

        print("Training Done")

    def makeCopy(self):
        torch.save(self.model, self.modelPath)

    def loadBestModel(self):
        return torch.load(
            self.modelPath,
            weights_only = False
        )