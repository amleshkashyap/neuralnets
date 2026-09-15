import torch
import matplotlib.pyplot as plt


class Evaluate:
    def __init__(self, model):
        self.model = model

    def evaluate(self, closePricesTest, indicatorsTest, priceDiffTest):
        self.model.eval()
        with torch.no_grad():
            trades = self.model(closePricesTest, indicatorsTest)
            trades = torch.round(trades * 100) / 100
            absReturn = torch.mul(trades, priceDiffTest)
            cumSumReturn = [0] + torch.cumsum(absReturn, dim = 0).view(-1).tolist()
            # buy and hold strategy
            cumSumPrice = [0] + torch.cumsum(priceDiffTest, dim = 0).view(-1).tolist()
            plt.title('Trading Evaluation On 2020')
            plt.plot(cumSumReturn, label = 'Model Returns')
            plt.plot(cumSumPrice, label = 'Buy And Hold Returns')
            plt.axhline(
                y = 0,
                color = 'black',
                linestyle = '--'
            )
            plt.legend()
            plt.savefig('evaluation.png')
            plt.show()
            print(f'Model Returns: {round(cumSumReturn[-1], 4)}')
            print(f'Buy And Hold Returns: {round(cumSumPrice[-1], 4)}')