import nni
import torch
from Parameters import *
from TcnClassifier import TcnClassifier
from Preprocess import Preprocess
from Train import Train

params = nni.get_next_parameter()
tclNum = params['tclNum']
tclChannelSize = params['tclChannelSize']
kernelSize = params['kernelSize']
dropout = params['dropout']
slices = params['slices']
useBias = params['useBias']
lr = params['lr']

# tclNum = 2
# tclChannelSize = 32
# kernelSize = 7
# dropout = 0.1
# slices = 1
# useBias = True
# lr = 0.005

channelSizes = [tclChannelSize] * tclNum

categoryData = {
    "RainToday": {
        "Yes": 1,
        "No": 0
    }
}

preprocess = Preprocess(
    None,
    categoryData
)

data = preprocess.getFullData()
preprocess.prepareData(data, INPUT_SIZE)

print("Data Prepared")

XTrain = preprocess.getXTrain()
XValidate = preprocess.getXValidate()
YTrain = preprocess.getYTrain()
YValidate = preprocess.getYValidate()

modelParams = {
    'numInputs': INPUT_SIZE,
    'numClasses': 2,
    'numChannels': channelSizes,
    'kernelSize': kernelSize,
    'dropout': dropout,
    'slices': slices,
    'act': 'relu',
    'useBias': useBias
}

model = TcnClassifier(**modelParams)
optimizer = torch.optim.Adam(
    params = model.parameters(),
    lr = lr
)

criterion = torch.nn.CrossEntropyLoss()
trainer = Train(model, criterion, optimizer)
minLoss = trainer.train(XTrain, XValidate, YTrain, YValidate, EPOCHS)
trainer.plotTraining()
print(minLoss)
nni.report_final_result(minLoss)