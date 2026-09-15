import os
import random
import queue
import numpy as np
import torch
from Evaluation import Evaluation
from Preprocess import Preprocess
from TCN import TCN
from Train import Train
from RMSLELoss import RMSLELoss
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import polars as pl
from kaggle.grocerySales.ReturnableThread import ReturnableThread
import time

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

FEATURES = 1
EPOCHS = 100
TEST_LENGTH = 100
CHANNELS = [10] * 4
KERNEL_SIZE = 5
DROPOUT = 0.
INPUT_SIZE = 4


MODEL_PATH = 'tmpModels/modelTmp.pt'
TRAIN_PATH = './../data/train/'
TEST_PATH = './../data/test/'


def runFamilyModel(familyTrainDf, familyTestDf, storeFamily):
    SCALER1 = StandardScaler()
    SCALER2 = MinMaxScaler()

    preprocess = Preprocess(FEATURES, TRAIN_PATH, SCALER2)
    preprocess.prepareData(familyTrainDf, TEST_LENGTH, 0.9)
    print(f"\nCompleted Data Preparation: {storeFamily}")

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

    modelPath = f'{os.getcwd()}\\tmp\\model_{storeFamily}.pt'

    trainer = Train(model, criterion, modelPath)
    print(f"\nStarting Training: {storeFamily}")

    trainer.train(XTrain, XValidate, YTrain, YValidate, EPOCHS, storeFamily)
    # trainer.plotTraining()

    evaluation = Evaluation(TCN(**modelParams), modelPath, criterion)
    evaluation.evaluate(XTest, YTest)
    print(f"\nEvaluated On Test Set: {storeFamily}")

    preprocessTest = Preprocess(FEATURES, TEST_PATH, scaler, 'test', categoryData)
    preprocessTest.prepareData(familyTestDf, 0)
    finalXTest = preprocessTest.getXTrain()
    finalYTest = preprocessTest.getYTrain()
    evaluation.evaluate(finalXTest, finalYTest, 'test')

    ids = preprocessTest.getIds()
    res = evaluation.getResults()
    results = pl.DataFrame()
    results = results.with_columns(
        pl.Series(
            name = 'id',
            values = list(ids),
            dtype = pl.Int64
        ),
        pl.Series(
            name = 'sales',
            values = res,
            dtype = pl.Float64
        )
    )
    return results


if __name__ == "__main__":
    stores = np.arange(1, 55)
    # stores = np.arange(1, 5)
    families = ['GROCERY I', 'HARDWARE', 'HOME AND KITCHEN II', 'LINGERIE', 'HOME APPLIANCES', 'CLEANING', 'PERSONAL CARE', 'MAGAZINES', 'DELI', 'PET SUPPLIES', 'SEAFOOD', 'FROZEN FOODS', 'PLAYERS AND ELECTRONICS', 'BEVERAGES', 'BOOKS', 'PREPARED FOODS', 'BABY CARE', 'CELEBRATION', 'GROCERY II', 'LAWN AND GARDEN', 'DAIRY', 'EGGS', 'BEAUTY', 'AUTOMOTIVE', 'HOME AND KITCHEN I', 'SCHOOL AND OFFICE SUPPLIES', 'LADIESWEAR', 'BREAD/BAKERY', 'PRODUCE', 'HOME CARE', 'MEATS', 'POULTRY', 'LIQUOR,WINE,BEER']
    # families = ['GROCERY I', 'HARDWARE', 'HOME AND KITCHEN II']
    finalRes: pl.DataFrame = pl.DataFrame()
    threads = []
    queue = queue.Queue()

    startTime = time.perf_counter()
    for store in stores:
        storeTrainDf = pl.read_csv(f'{TRAIN_PATH}/{store}.csv')
        storeTestDf = pl.read_csv(f'{TEST_PATH}/{store}.csv')
        count = 0
        for family in families:
            storeFamily = f"{store}_{count}"
            count += 1
            familyTrainDf = storeTrainDf.filter(pl.col('family') == family)
            familyTestDf = storeTestDf.filter(pl.col('family') == family)
            t = ReturnableThread(
                target = runFamilyModel,
                familyTrainDf = familyTrainDf,
                familyTestDf = familyTestDf,
                storeFamily = storeFamily,
                queue = queue)
            threads.append(t)
            if len(threads) == 10:
                results = []
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                    results.append(queue.get())
                for result in results:
                    finalRes = pl.concat([finalRes, result])
                threads = []

    endTime = time.perf_counter()
    print(f"Time taken: {round(endTime - startTime, 4)} seconds")
    finalRes.write_csv('resultsTCN.csv')