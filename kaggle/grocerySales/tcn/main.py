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
import polars as pl

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if __name__ == "__main__":
    FEATURES = 1
    EPOCHS = 100
    TEST_LENGTH = 1000
    CHANNELS = [10] * 4
    KERNEL_SIZE = 5
    DROPOUT = 0.

    INPUT_SIZE = 10

    SCALER = StandardScaler()

    MODEL_PATH = 'modelTmp.pt'
    TRAIN_PATH = './../data/train/'
    TEST_PATH = './../data/test/'

    stores = np.arange(1, 55)

    families = ['GROCERY I', 'HARDWARE', 'HOME AND KITCHEN II', 'LINGERIE', 'HOME APPLIANCES', 'CLEANING', 'PERSONAL CARE', 'MAGAZINES', 'DELI', 'PET SUPPLIES', 'SEAFOOD', 'FROZEN FOODS', 'PLAYERS AND ELECTRONICS', 'BEVERAGES', 'BOOKS', 'PREPARED FOODS', 'BABY CARE', 'CELEBRATION', 'GROCERY II', 'LAWN AND GARDEN', 'DAIRY', 'EGGS', 'BEAUTY', 'AUTOMOTIVE', 'HOME AND KITCHEN I', 'SCHOOL AND OFFICE SUPPLIES', 'LADIESWEAR', 'BREAD/BAKERY', 'PRODUCE', 'HOME CARE', 'MEATS', 'POULTRY', 'LIQUOR,WINE,BEER']

    finalRes: pl.DataFrame = pl.DataFrame()

    for store in stores:
        storeTrainDf = pl.read_csv(f'{TRAIN_PATH}/{store}.csv')
        storeTestDf = pl.read_csv(f'{TEST_PATH}/{store}.csv')
#        for family in families:
#            familyTrainDf = storeTrainDf.filter(pl.col('family') == family)
#            familyTestDf = storeTestDf.filter(pl.col('family') == family)

        preprocess = Preprocess(FEATURES, TRAIN_PATH, SCALER)
        preprocess.prepareData(storeTrainDf, TEST_LENGTH, 0.9)
        print("\nCompleted Data Preparation.")

        scaler = preprocess.getScaler()
        XTrain = preprocess.getXTrain()
        XValidate = preprocess.getXValidate()
        XTest = preprocess.getXTest()
        YTrain = preprocess.getYTrain()
        YValidate = preprocess.getYValidate()
        YTest = preprocess.getYTest()
        categoryData = preprocess.getCategoryData()

        print(XTrain[:500])
        print(YTrain[:500])

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
        print(f"\nStarting Training For Store: {store}")

        trainer.train(XTrain, XValidate, YTrain, YValidate, EPOCHS)
        # trainer.plotTraining()

        evaluation = Evaluation(TCN(**modelParams), MODEL_PATH, criterion)
        evaluation.evaluate(XTest, YTest)
        print("\nEvaluated On Test Set.")

        preprocessTest = Preprocess(FEATURES, TEST_PATH, scaler, 'test', categoryData)
        preprocessTest.prepareData(storeTestDf, 0)
        finalXTest = preprocessTest.getXTrain()
        finalYTest = preprocessTest.getYTrain()
        evaluation.evaluate(finalXTest, finalYTest, 'test')

        ids = preprocessTest.getIds()
        results = pl.DataFrame(
            pl.Series(
                name='id',
                values=list(ids),
                dtype=pl.Int64
            )
        )
        results = pl.concat([results, evaluation.getResults()], how = "horizontal")
        if len(finalRes) == 0:
            finalRes = results
        else:
            finalRes = pl.concat([finalRes, results])

    finalRes.write_csv('resultsTCN.csv')