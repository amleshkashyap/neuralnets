import torch

# simply selects the last value to make a prediction
class DummyPredictor(torch.nn.Module):
    def forward(self, x):
        lastValues = []
        for value in x.tolist():
            lastValues.append(value[-1])
        return torch.tensor(data = lastValues)