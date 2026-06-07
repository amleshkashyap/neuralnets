import torch
import torch.nn as nn

class Encode(nn.Module):
    def __init__(self, channels, embeddings, kernel):
        super(Encode, self).__init__()

        self.relu = nn.ReLU()
        self.filters = [32, 64, 128, 256, 512]

        self.conv1 = nn.Conv2d(channels, self.filters[0], kernel_size = kernel)
        self.bn1 = nn.BatchNorm2d(self.filters[0])

        self.conv2 = nn.Conv2d(self.filters[0], self.filters[1], kernel_size = kernel, stride = 2)
        self.bn2 = nn.BatchNorm2d(self.filters[1])

        self.conv3 = nn.Conv2d(self.filters[1], self.filters[2], kernel_size = kernel)
        self.bn3 = nn.BatchNorm2d(self.filters[2])

        self.conv4 = nn.Conv2d(self.filters[2], self.filters[3], kernel_size = kernel, stride = 2)
        self.bn4 = nn.BatchNorm2d(self.filters[3])

        self.conv5 = nn.Conv2d(self.filters[3], self.filters[4], kernel_size = kernel)
        self.bn5 = nn.BatchNorm2d(self.filters[4])

        self.mu = nn.Linear(self.filters[4] * 4, embeddings)
        self.sigma = nn.Linear(self.filters[4] * 4, embeddings)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.bn4(self.relu(self.conv4(x)))
        x = self.bn5(self.relu(self.conv5(x)))
        x = torch.flatten(x, start_dim = 1)

        return self.mu(x), self.sigma(x)