import torch
import torch.nn as nn

class TrainVAE:
    def __init__(self, X_train, model, device):
        self.device = device
        self.X_train = torch.from_numpy(X_train).to(self.device)

        self.model = model

        self.learning_rate = 0.001
        self.batch_size = 32

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        self.criterion = nn.MSELoss(reduction='sum')
        self.model.to(self.device)

    def train(self, epochs = 100):
        train_loader = torch.utils.data.DataLoader(
            self.X_train,
            batch_size = self.batch_size,
            shuffle = True
        )

        for epoch in range(epochs):
            total_loss = 0.0

            for batch_idx, data in enumerate(train_loader):
                data = data.to(self.device)

                encoded, decoded, mean, log_var = self.model(data)

                KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                loss = self.criterion(decoded, data) + 3 * KLD
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * data.size(0)

            epoch_loss = total_loss / len(train_loader.dataset)
            print("Epoch {}/{}: loss={:.4f}".format(epoch + 1, epochs, epoch_loss))

        return self.model