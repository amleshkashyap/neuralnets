from Preprocess import Preprocess
from Parameters import *
from TcnClassifier import TcnClassifier
from Evaluate import Evaluate
import torch

# fill with actual best params using NNI
modelParams = {
    'numInputs': INPUT_SIZE,
    'numClasses': 2,
    'numChannels': [32] * 2,
    'kernelSize': 7,
    'dropout': 0.1,
    'slices': 1,
    'act': 'relu',
    'useBias': True
}

model = TcnClassifier(**modelParams)
model.load_state_dict(
    torch.load(
        'model.pt',
        weights_only = True
    )
)

categoryData = {
    "RainToday": {
        "Yes": 1,
        "No": 0
    }
}

preprocessTest = Preprocess(
    None,
    categoryData,
    'test'
)

testData = preprocessTest.getFullData()
preprocessTest.prepareData(testData, INPUT_SIZE)

XTest = preprocessTest.getXTrain()
YTest = preprocessTest.getYTrain()

evaluate = Evaluate(model)
evaluate.evaluate(XTest, YTest)