import torch
from torch import optim
import torch.nn as nn


class TrainAE:
    def __init__(self, X_train, model):
        self.X_train = torch.from_numpy(X_train)

        # pass AutoEncoder()
        self.model = model

        self.device = 'CPU'
        self.learning_rate = 0.001
        self.batch_size = 32

        self.optimizer = optim.Adam(model.parameters(), lr = self.learning_rate)

        # define loss function
        self.criterion = nn.MSELoss()

        # move the model to the device
        self.model.to(self.device)

    def train(self, epochs = 100):
        # create dataloader to handle batching of training data
        train_loader = torch.utils.data.DataLoader(
            self.X_train,
            batch_size = self.batch_size,
            shuffle = True,
        )

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, data in enumerate(train_loader):
                # get a batch of data and move it to the selected device
                data = data.to(self.device)

                encoded, decoded = self.model(data)

                # compute loss and perform backpropagation
                loss = self.criterion(decoded, data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update loss
                total_loss += loss.item() * data.size(0)

            epoch_loss = total_loss / len(train_loader.dataset)
            print("Epoch {}/{}: loss={:.4f}".format(epoch + 1, epochs, epoch_loss))

        return self.model