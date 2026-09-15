from Preprocess import Preprocess
from StockTrader import StockTrader
from collections import OrderedDict
import torch
from stats.applications.stocks.Evaluate import Evaluate
import random
import numpy as np

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

rnnType = 'rnn'
rnnHiddenSize = 24
indicatorHiddenSize = 1
decisionSize = 2
ind1 = {
    '_name': 'rsi',
    'length': 20
}
ind2 = {
    '_name': 'cmo',
    'length': 20
}

START_DATE = '2020-01-01'
END_DATE = '2021-01-01'
WINDOW = 40

preprocess = Preprocess(None, 'test')

data = preprocess.getDataTill2021()
data = data[data.index < END_DATE]
tsLen = data[data.index > START_DATE].shape[0]

dataSource = OrderedDict()
dataSource['close_diff'] = (data['Close'] - data['Close'].shift(1))
dataSource['close_roc'] = (data['Close'] / data['Close'].shift(1))
dataSource['ind1'] = preprocess.getIndicator(data, ind1['_name'], ind1)
dataSource['ind2'] = preprocess.getIndicator(data, ind2['_name'], ind2)

for k, v in dataSource.items():
    dataSource[k] = v[v.index > START_DATE].dropna().values

dataSource['close_diff'][0] = 0
dataSource['close_roc'][0] = 1

D = []
for i in range(tsLen):
    row = []
    for k, v in dataSource.items():
        row.append(v[i])
    D.append(row)


preprocess.prepareData(D, WINDOW, 0)

XTrain = preprocess.getXTrain()
YTrain = preprocess.getYTrain()

closed = XTrain[:, :, :2]
indicator = XTrain[:, -1, 2:]
tomorrowPriceDiff = YTrain[:, :, 0].view(-1)

modelParams = {
    'rnnInputSize': 2,
    'indicatorInputSize': 2,
    'rnnType': rnnType,
    'rnnHiddenSize': rnnHiddenSize,
    'indicatorHiddenSize': indicatorHiddenSize,
    'decisionSize': decisionSize
}

model = StockTrader(**modelParams)
model.load_state_dict(torch.load(
   'model.pt',
    # 'best_model.pth',
    weights_only = True
))

evaluate = Evaluate(model)
evaluate.evaluate(closed, indicator, tomorrowPriceDiff)