import pandas as pd
import numpy as np
import os
import polars as pl
import torch


class Preprocess:
    def __init__(self, features, filePath, scaler, mode = 'train', categoryData = {}):
        self.filePath = filePath
        self.mode = mode
        self.scaler = scaler
        self.features = features
        self.data = None
        self.sales = None
        self.XTrain = None
        self.XValidate = None
        self.XTest = None
        self.YTrain = None
        self.YValidate = None
        self.YTest = None
        self.categoryData = categoryData

    def reset(self):
        self.data = None
        self.sales = None
        self.XTrain = None
        self.XValidate = None
        self.XTest = None
        self.YTrain = None
        self.YValidate = None
        self.YTest = None

    def getXTrain(self):
        return self.XTrain

    def getXValidate(self):
        return self.XValidate

    def getXTest(self):
        return self.XTest

    def getYTrain(self):
        return self.YTrain

    def getYValidate(self):
        return self.YValidate

    def getYTest(self):
        return self.YTest

    def getCategoryData(self):
        return self.categoryData

    def getScaler(self):
        return self.scaler

    def slidingWindow(self, totalLength):
        X = []
        Y = []

        # for i in range(self.features + 1, len(self.data) + 1):
        for i in range(self.features + 1, totalLength + 1):
            X.append(self.data[i - (self.features + 1): i - 1])
            Y.append(self.sales[i - 1])

        return X, Y

    def loadMergedData(self):
        self.data = pl.read_csv(self.filePath)
        self.data.sort('date')
        self.data.drop_in_place('id')
        self.data.drop_in_place('transactions')
        self.data.drop_in_place('holiday_description')
        if self.mode == 'train':
            self.sales = self.data.get_column('sales')
            self.data.drop_in_place('sales')
        else:
            self.sales = torch.zeros(len(self.data))

    def cleanupMergedData(self):
        convertToNumber = [
            'date',
            'family',
            # 'holiday_description',
            'city',
            'state',
            'store_type',
            'holiday_type'
        ]
        self.data = Preprocess.convertToNumber(self.data, convertToNumber, self.categoryData)
        fillMissingCols = list(set(self.data.columns) - {'id', 'store_nbr'})
        self.data = Preprocess.fillMissingValues(self.data, fillMissingCols)


    def prepareData(self, totalLength, testLength, trainRatio = 0.7):
        self.reset()
        self.loadMergedData()
        self.cleanupMergedData()
        print("\nCleanup Completed")
        X, Y = self.slidingWindow(totalLength)
        print("\nSliding Window Completed")
        if self.mode == 'train':
            trainLength = round(len(self.data) * trainRatio)
            trainLength = round(totalLength * trainRatio)


            self.XTrain, self.YTrain, self.XTest, self.YTest = \
                X[0:-testLength], Y[0:-testLength], X[-testLength:], Y[-testLength:]

            self.XTrain, self.YTrain, self.XValidate, self.YValidate = \
                self.XTrain[0:trainLength], self.YTrain[0:trainLength], \
                    self.XTrain[trainLength:], self.YTrain[trainLength:]

            xTrain = np.array(self.XTrain)
            shape = xTrain.shape
            data2d = xTrain.reshape(-1, shape[-1])
            scaled = self.scaler.fit_transform(data2d)
            self.XTrain = scaled.reshape(shape)

            self.XTrain = torch.tensor(
                data = self.XTrain
            ).float().transpose(1, 2)

            self.XValidate = torch.tensor(
                data = self.getScaled(self.XValidate)
            ).float().transpose(1, 2)

            self.XTest = torch.tensor(
                data = self.getScaled(self.XTest)
            ).float().transpose(1, 2)

            self.YTrain = torch.tensor(
                data = np.array(self.YTrain)
            ).float()

            self.YValidate = torch.tensor(
                data = np.array(self.YValidate)
            ).float()

            self.YTest = torch.tensor(
                data = np.array(self.YTest)
            ).float()
        else:
            self.XTrain = torch.tensor(
                data = self.getScaled(X)
            ).float().transpose(1, 2)

            self.YTrain = torch.tensor(
                data = np.array(Y)
            ).float()

    def getScaled(self, data):
        data = np.array(data)
        shape = data.shape
        data2d = data.reshape(-1, shape[-1])
        scaled = self.scaler.transform(data2d)
        data = scaled.reshape(shape)
        return data

    @staticmethod
    def groupDfByColumn(df: pl.DataFrame, columnName: str, value):
        return df.filter(pl.col(columnName) == value)

    @staticmethod
    def writeDfToCsv(df: pl.DataFrame, fileName: str):
        df.write_csv(fileName)

    @staticmethod
    def convertToNumber(df: pl.DataFrame, convertToNumeric: list[str], categoryData: dict):
        for col in convertToNumeric:
            objects = df[col].to_list()
            if categoryData.get(col) == None:
                categories = {}
            else:
                categories = categoryData.get(col)
            count = 1

            for category in objects:
                if categories.get(category) is None:
                    categories[category] = count
                    count += 1

            categoryData[col] = categories
            for i in range(len(objects)):
                if type(objects[i]) is not str:
                    continue
                objects[i] = categories[objects[i]]

            df = df.with_columns(
                pl.Series(
                    name = col,
                    values = objects,
                    dtype = pl.Int64
                )
            )

        return df

    @staticmethod
    def fillMissingValues(df, fillMissing):
        for col in fillMissing:
            objects = df[col]
            df = df.with_columns(
                pl.Series(
                    name = col,
                    values = objects,
                )
            )
            objects = df[col]
            df = df.with_columns(
                pl.col(col).fill_nan(objects.median()),
            )
            objects = df[col]
            df = df.with_columns(
                pl.col(col).fill_null(objects.median()),
            )
            objects = df[col]
            df = df.with_columns(
                pl.Series(
                    name = col,
                    values = objects,
                )
            )
        return df