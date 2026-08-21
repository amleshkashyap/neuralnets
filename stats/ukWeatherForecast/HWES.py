import torch
import torch.nn as nn
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class HWES(nn.Module):
    def forward(self, x):
        lastValues = []
        for value in x.tolist():
            model = ExponentialSmoothing(
                value,
                trend = None,
                seasonal = 'add',
                seasonal_periods = 12
            )
            results = model.fit()
            lastValues.append(results.forecast()[0])
        return torch.tensor(data = lastValues)