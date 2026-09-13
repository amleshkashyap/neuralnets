import os
from typing import Any
import torch
import matplotlib.pyplot as plt
import polars as pl
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np


class Preprocess:
    def __init__(self, scaler, categoryData: dict[Any, Any], mode = 'train'):
        self.basePath = os.getcwd()
        self.XTrain = None
        self.XValidate = None
        self.YTrain = None
        self.YValidate = None
        self.mode = mode
        self.scaler = scaler
        self.categoryData = categoryData
        self.convertToNumber = ['RainToday']
        self.fillMissing = [
            'MinTemp',
            'MaxTemp',
            'Rainfall',
            'WindGustSpeed',
            'WindSpeed9am',
            'WindSpeed3pm',
            'Humidity9am',
            'Humidity3pm',
            'Pressure9am',
            'Pressure3pm',
            'Cloud9am',
            'Cloud3pm',
            'Temp9am',
            'Temp3pm',
            'RainToday'
        ]

    def reset(self):
        self.XTrain = None
        self.XValidate = None
        self.YTrain = None
        self.YValidate = None

    def getXTrain(self):
        return self.XTrain

    def getXValidate(self):
        return self.XValidate

    def getYTrain(self):
        return self.YTrain

    def getYValidate(self):
        return self.YValidate

    def getCategoryData(self):
        return self.categoryData

    # target length is 1
    def slidingWindow(self, data, window):
        X = []
        Y = []
        for i in range(window + 1, len(data) + 1):
            X.append(data[i - (window + 1):i - 1])
            Y.append(data[i - 1])
        return X, Y

    def getDataTill2016(self):
        return pl.read_csv(f'{self.basePath}/data/weatherAUS_until_2016_01_01.csv').with_columns(
            pl.col('Date').str.to_date(format = '%Y-%m-%d')
        )

    def getFullData(self):
        return pl.read_csv(f'{self.basePath}/data/weatherAUS_complete.csv').with_columns(
            pl.col('Date').str.to_date(format = '%Y-%m-%d')
        )

    def visualizeRainfall(self, df: pl.DataFrame):
        dfSydney: pl.DataFrame = df.filter(
            pl.col('Location') == 'Sydney'
        )
        dfSydney = dfSydney.filter(
            pl.col('Date') > datetime.strptime("2015-01-01", "%Y-%m-%d")
        )
        rainfall = dfSydney['Rainfall'].to_list()
        raintoday = dfSydney['RainToday'].map_elements(lambda x: 1 if x == 'Yes' else 0).to_list()
        fig, axs = plt.subplots(2)
        axs[0].set_title('Rainfall')
        axs[0].plot(rainfall)
        axs[1].set_title('Rain Classification')
        axs[1].bar(range(len(raintoday)), raintoday, color = 'red')
        plt.savefig('raintoday.png')
        plt.show()

    def cleanupData(self, data: pl.DataFrame, window):
        data = Preprocess.convertToNumber(data, self.convertToNumber, self.categoryData)
        locations = ['Albury', 'Newcastle', 'Richmond', 'Sydney', 'Canberra']
        newData = pl.DataFrame()
        X, Y = [], []
        cols = data.columns

        for location in locations:
            dfLoc = data.filter(pl.col('Location') == location)
            for col in list(set(cols) - set(self.fillMissing)):
                dfLoc.drop_in_place(col)

            dfLoc = Preprocess.fillMissingValues(dfLoc, self.fillMissing)
            pl.concat([newData, dfLoc])
            inp, output = self.slidingWindow(dfLoc, window)
            rain = [r.get_column(r.columns[-1]).to_list()[0] for r in output]
            X.extend(inp)
            Y.extend(rain)

        self.data = newData
        return X, Y

    def cleanupDataTest(self, data: pl.DataFrame, window):
        data = Preprocess.convertToNumber(data, self.convertToNumber, self.categoryData)
        X, Y = [], []
        cols = data.columns

        dfSydney = data.filter(pl.col('Location') == 'Sydney')
        dfSydney = dfSydney.filter(
            pl.col('Date') > datetime.strptime("2015-12-17", "%Y-%m-%d")
        )
        dfSydney = dfSydney.filter(
            pl.col('Date') < datetime.strptime("2017-01-01", "%Y-%m-%d")
        )
        for col in list(set(cols) - set(self.fillMissing)):
            dfSydney.drop_in_place(col)

        dfSydney = Preprocess.fillMissingValues(dfSydney, self.fillMissing)
        inp, output = self.slidingWindow(dfSydney, window)
        rain = [r.get_column(r.columns[-1]).to_list()[0] for r in output]
        X.extend(inp)
        Y.extend(rain)

        self.data = dfSydney
        return X, Y

    def prepareData(self, data, window):
        self.reset()
        if self.mode == 'train':
            X, Y = self.cleanupData(data, window)
            self.XTrain, self.XValidate, self.YTrain, self.YValidate = train_test_split(
                X,
                Y,
                test_size = 0.3,
                shuffle = False
            )
            self.XTrain = torch.tensor(
                data = np.array(self.XTrain)
            ).float().transpose(1, 2)

            self.XValidate = torch.tensor(
                data = np.array(self.XValidate)
            ).float().transpose(1, 2)

            self.YTrain = torch.tensor(
                data = np.array(self.YTrain)
            ).long()
            self.YValidate = torch.tensor(
                data = np.array(self.YValidate)
            ).long()
        else:
            X, Y = self.cleanupDataTest(data, window)
            self.XTrain = torch.tensor(
                data = np.array(X)
            ).float().transpose(1, 2)
            self.YTrain = torch.tensor(
                data = np.array(Y)
            ).long()


    @staticmethod
    def convertToNumber(df, convertToNumeric, categoryData):
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
            median = objects.median()
            if median is None:
                median = 0.0
            df = df.with_columns(
                pl.col(col).fill_nan(value = median),
            )
            objects = df[col]
            median = objects.median()
            if median is None:
                median = 0.0
            df = df.with_columns(
                pl.col(col).fill_null(value = median),
            )
            objects = df[col]
            df = df.with_columns(
                pl.Series(
                    name = col,
                    values = objects,
                )
            )
        return df

if __name__ == "__main__":
    p = Preprocess()
    data = p.getDataTill2016()
    p.visualizeRainfall(data)