import torch.nn as nn
import torch

class StockTrader(nn.Module):
    def __init__(self,
                 rnnInputSize,
                 indicatorInputSize,
                 rnnType = 'gru',
                 rnnHiddenSize = 16,
                 indicatorHiddenSize = 4,
                 decisionSize = 4
                 ):
        super(StockTrader, self).__init__()
        rnnParams = {
            'input_size': rnnInputSize,
            'hidden_size': rnnHiddenSize,
            'batch_first': True
        }
        if rnnType == 'gru':
            self.rnn = nn.GRU(**rnnParams)
        elif rnnType == 'rnn':
            self.rnn = nn.RNN(**rnnParams)
        else:
            raise Exception(f'Unsupported RNN Type: {rnnType}')

        self.rnnInputSize = rnnInputSize
        self.indicatorInputSize = indicatorInputSize

        self.linearIndicator = nn.Linear(indicatorInputSize, indicatorHiddenSize)
        self.linearDecision = nn.Linear(rnnHiddenSize + indicatorHiddenSize, decisionSize)
        self.linearPos = nn.Linear(decisionSize, 1)

        # use these variable names when loading state dict from best_model.pth
        # self.lin_ind = nn.Linear(indicatorInputSize, indicatorHiddenSize)
        # self.lin_des = nn.Linear(rnnHiddenSize + indicatorHiddenSize, decisionSize)
        # self.lin_pos = nn.Linear(decisionSize, 1)

    def forward(self, raw, indicators, rnnHidden = None):
        # RNN works on raw closing prices to identify latent features
        _, h = self.rnn(raw, rnnHidden)

        # Indicators are assumed to content latent information, hence passed through linear layer
        z = torch.relu(self.linearIndicator(indicators))
        # z = torch.relu(self.lin_ind(indicators))

        # concatenate the output from both models
        x = torch.cat((z, h[0]), dim = 1)

        # decision layer is a simple linear layer
        x = torch.relu(self.linearDecision(x))
        # x = torch.relu(self.lin_des(x))

        # tanh is used to normalize the result - number of shares to buy can be fractional during training
        p = torch.tanh(self.linearPos(x))
        # p = torch.tanh(self.lin_pos(x))
        return p.view(-1)