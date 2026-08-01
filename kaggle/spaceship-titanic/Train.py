
class Train:
    def __init__(self, model, dataloader, criterion, optimizer):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, epochs = 50):
        for epoch in range(epochs):
            epochLoss = 0.0
            for batchX, batchY in self.dataloader:
                # reset the gradients
                self.optimizer.zero_grad()

                # run the forward method
                y = self.model(batchX)

                # compute the loss values
                loss = self.criterion(y, batchY)
                loss.backward()
                self.optimizer.step()

                epochLoss += loss.item() * batchX.size(0)

            totalEpochLoss = epochLoss / len(self.dataloader.dataset)
            print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {totalEpochLoss:.4f}")

        print("Training Done")