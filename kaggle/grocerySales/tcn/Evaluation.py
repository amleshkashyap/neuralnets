import matplotlib.pyplot as plt
from Dummy import Dummy
import torch
import polars as pl

class Evaluation:
    def __init__(self, model, modelPath, criterion):
        model.load_state_dict(
            torch.load(modelPath, weights_only = True)
        )
        self.model = model
        self.criterion = criterion

    def evaluate(self, XTest, YTest, mode = 'train'):
        self.model.eval()
        tcnPrediction = self.model(XTest)
        if mode == 'train':
            dummyPrediction = Dummy()(XTest)
            tcnLoss = round(self.criterion(tcnPrediction, YTest).item(), 4)
            dummyLoss = round(self.criterion(dummyPrediction, YTest).item(), 4)

            plt.title(f'Test| TCN: {tcnLoss}; Dummy: {dummyLoss}')
            plt.plot(
                tcnPrediction.view(-1).tolist(),
                label = 'TCN'
            )
            plt.plot(
                YTest.tolist(),
                label = 'Real'
            )
            plt.legend()
            plt.savefig('evaluation.png')
            plt.show()
        else:
            finalDf = pl.DataFrame()
            finalDf = finalDf.with_columns(
                pl.Series(
                    name = 'sales',
                    values = list(tcnPrediction),
                    dtype = pl.Float32
                )
            )
            finalDf.write_csv('resultsTCN.csv')