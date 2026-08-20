from statsmodels.tsa.holtwinters import ExponentialSmoothing
import torch

class HwesPredictor(torch.nn.Module):
    def forward(self, x):
        lastValues = []
        for value in x.tolist():
            model = ExponentialSmoothing(value)
            results = model.fit()
            lastValues.append(results.forecast()[0])
        return torch.tensor(data = lastValues)