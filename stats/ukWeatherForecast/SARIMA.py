import torch
import torch.nn as nn
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMA(nn.Module):
    def forward(self, x):
        lastValues = []
        l = x.tolist()
        counter = 0
        for value in l:
            model = SARIMAX(
                value,
                order = (1, 1, 1),
                seasonal_order = (1, 1, 1, 12)
            )
            results = model.fit()
            lastValues.append(results.forecast()[0])
            counter += 1
            print(f'DEBUG: SARIMA calculation {counter} / {len(l)}')
        return torch.tensor(data = lastValues)