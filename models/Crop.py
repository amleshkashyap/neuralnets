import torch.nn as nn

class Crop(nn.Module):
    def __init__(self, cropSize):
        super(Crop, self).__init__()
        self.cropSize = cropSize

    def forward(self, x):
        return x[:, :, :-self.cropSize].contiguous()