import os
import random
import numpy as np
import torch
from Evaluation import Evaluation
from Preprocess import Preprocess
from TCN import TCN
from Train import Train
from RMSLELoss import RMSLELoss
from sklearn.preprocessing import MinMaxScaler, StandardScaler

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if __name__ == "__main__":
    FEATURES = 1
    EPOCHS = 100
    DATA_LENGTH = 200000
    TEST_LENGTH = 3000
    CHANNELS = [10] * 4
    KERNEL_SIZE = 5
    DROPOUT = 0.

    INPUT_SIZE = 10

    SCALER = StandardScaler()

    MODEL_PATH = 'modelTmp.pt'
    TRAIN_PATH = f'{os.getcwd()}/../trainMerged.csv'
    TEST_PATH = f'{os.getcwd()}/../testMerged.csv'

    preprocess = Preprocess(FEATURES, TRAIN_PATH, SCALER)
    preprocess.prepareData(DATA_LENGTH, TEST_LENGTH, 0.8)
    print("\nCompleted Data Preparation.")

    scaler = preprocess.getScaler()
    XTrain = preprocess.getXTrain()
    XValidate = preprocess.getXValidate()
    XTest = preprocess.getXTest()
    YTrain = preprocess.getYTrain()
    YValidate = preprocess.getYValidate()
    YTest = preprocess.getYTest()
    categoryData = preprocess.getCategoryData()

    modelParams = {
        'inputSize': INPUT_SIZE,
        'outputSize': 1,
        'channels': CHANNELS,
        'kernelSize': KERNEL_SIZE,
        'dropout': DROPOUT
    }

    model = TCN(**modelParams)
    criterion = RMSLELoss()
    print("\nLoaded Model.")

    trainer = Train(model, criterion, MODEL_PATH)
    print("\nStarting Training.")

    trainer.train(XTrain, XValidate, YTrain, YValidate, EPOCHS)
    trainer.plotTraining()

    evaluation = Evaluation(TCN(**modelParams), MODEL_PATH, criterion)
    evaluation.evaluate(XTest, YTest)
    print("\nEvaluated On Test Set.")

    preprocessTest = Preprocess(FEATURES, TEST_PATH, scaler, 'test', categoryData)
    preprocessTest.prepareData(28512, 0)
    finalXTest = preprocessTest.getXTrain()
    finalYTest = preprocessTest.getYTrain()
    evaluation.evaluate(finalXTest, finalYTest, 'test')