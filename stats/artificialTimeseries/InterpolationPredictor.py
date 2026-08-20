from scipy import interpolate
import torch
import numpy as np
import random
import torch

random.seed(1)
torch.manual_seed(1)

class InterpolationPredictor(torch.nn.Module):
    def forward(self, x):
        lastValues = []
        # value is of length windowLen
        for value in x.tolist():
            x1 = np.arange(0, len(value))
            y = interpolate.interp1d(
                x1,
                value,
                fill_value = 'extrapolate'
            )
            lastValues.append([y(len(value)).tolist()])
        return torch.tensor(data = lastValues).squeeze(-1)