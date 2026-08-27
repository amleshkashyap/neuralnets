import nni
from Preprocessing import Preprocessing
from GRU import GRU
from Train import Train
from Evaluation import Evaluation
from sklearn.preprocessing import MinMaxScaler
import torch

FEATURES = 240
TEST_LENGTH = 100

trialParams = nni.get_next_parameter()

optimizerName = trialParams['optimizer']
gruHiddenSize = trialParams['gruHiddenSize']
learningRate = trialParams['learningRate']

EPOCHS = 50

preprocess = Preprocessing()
data = preprocess.getPjmeTimeseries()
scaler = MinMaxScaler()
scaledData = scaler.fit_transform(data)
preprocess.prepareData(scaledData, FEATURES, TEST_LENGTH)

XTrain = preprocess.getXTrain()
XValidate = preprocess.getXValidate()
XTest = preprocess.getXTest()

YTrain = preprocess.getYTrain()
YValidate = preprocess.getYValidate()
YTest = preprocess.getYTest()

model = GRU(
    1,
    gruHiddenSize,
    1
)

optimizers = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    'adamax': torch.optim.Adamax
}

optimizer = optimizers[optimizerName](
    params = model.parameters(),
    lr = learningRate
)

criterion = torch.nn.MSELoss()

trainer = Train(model, optimizer, criterion)
trainer.train(XTrain, XValidate, YTrain, YValidate, EPOCHS)

bestModel = trainer.getBestModel()

evaluation = Evaluation(bestModel, criterion)
evaluation.evaluate(XValidate, XTest, YTest)