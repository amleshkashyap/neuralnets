import torch.nn as nn

class Decode(nn.Module):
    def __init__(self, channels, embeddings, kernel):
        super(Decode, self).__init__()
        self.relu = nn.ReLU()
        # NOTE:
        self.filters = [512, 256, 128, 64, 32]

        # NOTE:
        self.decodeIn = nn.Linear(embeddings, self.filters[0] * 4)

        # NOTE:
        self.deconv1 = nn.ConvTranspose2d(self.filters[0], self.filters[1], kernel_size = kernel, stride = 2, dilation = 1, output_padding = 1, padding = 1)
        # NOTE:
        self.bn1 = nn.BatchNorm2d(self.filters[1])

        self.deconv2 = nn.ConvTranspose2d(self.filters[1], self.filters[2], kernel_size = kernel, stride = 2, dilation = 1, output_padding = 0, padding = 1)
        self.bn2 = nn.BatchNorm2d(self.filters[2])

        self.deconv3 = nn.ConvTranspose2d(self.filters[2], self.filters[3], kernel_size = kernel, stride = 2, dilation = 1, output_padding = 1, padding = 1)
        self.bn3 = nn.BatchNorm2d(self.filters[3])

        self.deconv4 = nn.ConvTranspose2d(self.filters[3], self.filters[4], kernel_size = kernel, stride = 2, dilation = 1, output_padding = 0, padding = 1)
        self.bn4 = nn.BatchNorm2d(self.filters[4])

        # NOTE:
        self.decodeOut = nn.Conv2d(self.filters[4], 1, kernel_size = kernel, padding = 1)

    def forward(self, x):
        x = self.decodeIn(x)
        # NOTE:
        x = x.view(-1, 512, 2, 2)
        x = self.bn1(self.relu(self.deconv1(x)))
        x = self.bn2(self.relu(self.deconv2(x)))
        x = self.bn3(self.relu(self.deconv3(x)))
        x = self.bn4(self.relu(self.deconv4(x)))
        x = self.decodeOut(x)

        return x