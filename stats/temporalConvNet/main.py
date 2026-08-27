import random
import numpy as np
import torch
from Evaluation import Evaluation
from Preprocess import Preprocess
from TCN import TCN
from Train import Train

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if __name__ == "__main__":
    FEATURES = 20
    EPOCHS = 1000
    DATA_LENGTH = 5000
    TEST_LENGTH = 300
    CHANNELS = [10] * 4
    KERNEL_SIZE = 5
    DROPOUT = 0.

    preprocess = Preprocess()
    originalData = preprocess.generateTimeseries(DATA_LENGTH)
    data = preprocess.getDifferencedData(originalData)
    preprocess.prepareData(data, FEATURES, TEST_LENGTH)

    XTrain = preprocess.getXTrain()
    XValidate = preprocess.getXValidate()
    XTest = preprocess.getXTest()
    YTrain = preprocess.getYTrain()
    YValidate = preprocess.getYValidate()
    YTest = preprocess.getYTest()

    trainLength = XTrain.size()[0]

    modelParams = {
        'inputSize': 4,
        'outputSize': 1,
        'channels': CHANNELS,
        'kernelSize': KERNEL_SIZE,
        'dropout': DROPOUT
    }

    model = TCN(**modelParams)
    criterion = torch.nn.MSELoss()

    trainer = Train(model, criterion)

    trainer.train(XTrain, XValidate, YTrain, YValidate, EPOCHS)
    trainer.plotTraining()
    bestModel = trainer.loadBestModel()

    evaluation = Evaluation(bestModel, criterion, preprocess, originalData, TEST_LENGTH)
    evaluation.evaluate(XTest, YTest)