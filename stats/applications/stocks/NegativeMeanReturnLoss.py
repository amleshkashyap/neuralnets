import torch.nn as nn
import torch

class NegativeMeanReturnLoss(nn.Module):
    def __init__(self):
        super(NegativeMeanReturnLoss, self).__init__()

    # loss functions tend to minimize the loss, but we've to maximize a metric (which is
    #   shares to buy * price difference) - hence, we return the negative
    def forward(self, shares, priceDiff):
        absReturn = torch.mul(shares.view(-1), priceDiff)
        ar = torch.mean(absReturn)
        return torch.neg(ar)