from collections import OrderedDict
import torch
from Preprocess import Preprocess
from StockTrader import StockTrader
from NegativeMeanReturnLoss import NegativeMeanReturnLoss
from Train import Train
import pandas_ta_classic as ta
import random
import numpy as np

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

# params = nni.get_next_parameter()

# lr = params['lr']
# rnnType = params['rnnType']
# rnnHiddenSize = params['rnnHiddenSize']
# indicatorHiddenSize = params['indicatorHiddenSize']
# decisionSize = params['decisionSize']
# ind1Name = params['ind1']['_name']
# ind2Name = params['ind2']['_name']

lr = 0.01
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

START_DATE = '2010-01-01'
END_DATE = '2020-01-01'
WINDOW = 40
EPOCHS = 500
TRAIN_RATIO = 0.8

preprocess = Preprocess(None)

data = preprocess.getDataTill2020()
data = data[data.index < END_DATE]
tsLen = data[data.index > START_DATE].shape[0]
trainLen = int(tsLen * TRAIN_RATIO)

dataSource = OrderedDict()
dataSource['close_diff'] = (data['Close'] - data['Close'].shift(1))
dataSource['close_roc'] = (data['Close'] / data['Close'].shift(1))
dataSource['ind1'] = preprocess.getIndicator(data, ind1['_name'], ind1)
dataSource['ind2'] = preprocess.getIndicator(data, ind2['_name'], ind2)

for k, v in dataSource.items():
    dataSource[k] = v[v.index > START_DATE].dropna().values

D = []
for i in range(tsLen):
    row = []
    for k, v in dataSource.items():
        row.append(v[i])
    D.append(row)


preprocess.prepareData(D, WINDOW, trainLen)

XTrain = preprocess.getXTrain()
XValidate = preprocess.getXValidate()
YTrain = preprocess.getYTrain()
YValidate = preprocess.getYValidate()

ClosedTrain, ClosedValidate = XTrain[:, :, :2], XValidate[:, :, :2]
IndTrain, IndValidate = XTrain[:, -1, 2:], XValidate[:, -1, 2:]
PriceTrain, PriceValidate = YTrain[:, :, 0].view(-1), YValidate[:, :, 0].view(-1)

modelParams = {
    'rnnInputSize': 2,
    'indicatorInputSize': 2,
    'rnnType': rnnType,
    'rnnHiddenSize': rnnHiddenSize,
    'indicatorHiddenSize': indicatorHiddenSize,
    'decisionSize': decisionSize
}

model = StockTrader(**modelParams)
criterion = NegativeMeanReturnLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
trainer = Train(model, criterion, optimizer)
trainer.train(
    ClosedTrain,
    ClosedValidate,
    IndTrain,
    IndValidate,
    PriceTrain,
    PriceValidate,
    EPOCHS
)
trainer.plotTraining()