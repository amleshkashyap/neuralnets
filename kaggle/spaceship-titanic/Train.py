from Parameters import *

class Train:
    def __init__(self, model, dataloader, criterion, optimizer):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, epochs = EPOCHS):
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

            totalEpochLoss = epochLoss / len(self.dataloader.dataset)
            print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {totalEpochLoss:.4f}")

        print("Training Done")