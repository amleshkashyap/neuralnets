from Preprocess import Preprocess
from NeuralNet import NeuralNet
from SARIMA import SARIMA
from HWES import HWES
from Train import Train
from Evaluation import Evaluation
import random
import torch
import os

torch.manual_seed(1)
random.seed(1)


if __name__ == "__main__":
    FEATURES = 120

    preprocess = Preprocess()
    preprocess.rawMetrics()
    preprocess.plotInterpolatedData()
    preprocess.prepareData(FEATURES, 60)

    XTrain = preprocess.getXTrain()
    XValidate = preprocess.getXValidate()
    XTest = preprocess.getXTest()
    YTrain = preprocess.getYTrain()
    YValidate = preprocess.getYValidate()
    YTest = preprocess.getYTest()

    neuralNet = NeuralNet(
        FEATURES,
        400,
        48,
        6,
        36,
        12,
        0.1,
        1
    )
    sarima = SARIMA()
    hwes = HWES()

    optimizer = torch.optim.Adam(
        neuralNet.parameters(),
    )
    criterion = torch.nn.L1Loss()

    trainer = Train(
        neuralNet,
        optimizer,
        criterion
    )
    trainer.train(
        XTrain,
        XValidate,
        YTrain,
        YValidate,
        150
    )
    trainer.plotTrainingProgress()
    bestNeuralNet = trainer.loadBestModel()

    evaluator = Evaluation(
        bestNeuralNet,
        sarima,
        hwes,
        criterion
    )
    evaluator.evaluateOnTest(
        XTest,
        YTest,
    )

    evaluator.checkAgainstActual(YTest)
    evaluator.checkStdDev(YTest)