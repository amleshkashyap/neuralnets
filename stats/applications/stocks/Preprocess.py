import os
import pandas as pd
import pandas_ta_classic as ta
import matplotlib.pyplot as plt
import torch
import numpy as np

class Preprocess:
    def __init__(self, scaler, mode = 'train'):
        self.scaler = scaler
        self.mode = mode
        self.basePath = os.getcwd()
        self.data = None
        self.XTrain = None
        self.XValidate = None
        self.YTrain = None
        self.YValidate = None

    def reset(self):
        self.data = None
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

    def slidingWindow(self, data, window):
        X, Y = [], []
        for i in range(window + 1, len(data) + 1):
            X.append(data[i - (window + 1): i - 1])
            Y.append(data[i - 1: i])
        return X, Y

    def prepareData(self, data, window, trainLen):
        self.reset()
        X, Y = self.slidingWindow(data, window)
        if self.mode == 'train':
            self.XTrain, self.XValidate = X[:trainLen], X[trainLen:]
            self.YTrain, self.YValidate = Y[:trainLen], Y[trainLen:]

            self.XTrain = torch.tensor(
                data = np.array(self.XTrain)
            ).float()
            self.XValidate = torch.tensor(
                data = np.array(self.XValidate)
            ).float()
            self.YTrain = torch.tensor(
                data = np.array(self.YTrain)
            ).float()
            self.YValidate = torch.tensor(
                data = np.array(self.YValidate)
            ).float()
        else:
            self.XTrain = torch.tensor(
                data = np.array(X)
            ).float()
            self.YTrain = torch.tensor(
                data = np.array(Y)
            ).float()

    def getIndicator(self, data, indicatorName, params):
        ts = None
        if indicatorName == 'ao':
            ts = data.ta.ao(params['fast'], params['slow'])
        elif indicatorName == 'apo':
            ts = data.ta.apo(params['fast'], params['slow'])
        elif indicatorName == 'cci':
            ts = data.ta.cci(params['length']) / 100
        elif indicatorName == 'cmo':
            ts = data.ta.cmo(params['length']) / 100
        elif indicatorName == 'mom':
            ts = data.ta.mom(params['length'])
        elif indicatorName == 'rsi':
            ts = data.ta.rsi(params['length']) / 100
        elif indicatorName == 'tsi':
            ts = data.ta.tsi(params['fast'], params['slow']) / 100

        return ts

    def getDataTill2020(self) -> pd.DataFrame:
        return pd.read_csv(
            f'{self.basePath}/data/MSFT_until_2020_01_01.csv',
            date_format = '%Y-%m-%d',
            index_col = 'Date'
        )

    def getDataTill2021(self) -> pd.DataFrame:
        return pd.read_csv(
            f'{self.basePath}/data/MSFT_until_2021_01_01.csv',
            date_format = '%Y-%m-%d',
            index_col = 'Date'
        )

    def plotSingleMetric(self, metric = 'Close', year = 2020):
        if metric not in ['Open', 'Close', 'Volume', 'High', 'Low', 'Adj Close']:
            return
        if year not in [2020, 2021]:
            return

        if year == 2020:
            data = self.getDataTill2020()
        else:
            data = self.getDataTill2021()
        data[metric].plot()
        plt.savefig(f'analysis/data_{metric}_{year}.png')
        plt.show()

    def plotIndicators(self, metric = 'Close', year = 2020):
        if metric not in ['Open', 'Close', 'Volume', 'High', 'Low', 'Adj Close']:
            return
        if year not in [2020, 2021]:
            return

        if year == 2020:
            data = self.getDataTill2020()
        else:
            data = self.getDataTill2021()

        fig, axes = plt.subplots(nrows = 3)
        axes[0].set_title('Microsoft Quotes')
        data[metric].plot(ax = axes[0])
        axes[1].set_title('Momentum Indicator')
        data.ta.ao(9, 14).plot(ax = axes[1])
        axes[2].set_title('RSI Indicator')
        data.ta.rsi(20).plot(ax = axes[2])
        for i in range(3):
            axes[i].xaxis.set_visible(False)
        plt.savefig(f'analysis/indicators_{metric}_{year}.png')
        plt.show()

    def plotAbsoluteChange(self, metric = 'Close', year = 2020):
        if metric not in ['Open', 'Close', 'Volume', 'High', 'Low', 'Adj Close']:
            return
        if year not in [2020, 2021]:
            return

        if year == 2020:
            data = self.getDataTill2020()
        else:
            data = self.getDataTill2021()

        (data[metric] - data[metric].shift(1)).plot()
        plt.title(f'Microsoft Quotes Absolute Change: {metric}, {year}')
        plt.savefig(f'analysis/abschange_{metric}_{year}.png')
        plt.show()

    def plotRelativeChange(self, metric = 'Close', year = 2020):
        if metric not in ['Open', 'Close', 'Volume', 'High', 'Low', 'Adj Close']:
            return
        if year not in [2020, 2021]:
            return

        if year == 2020:
            data = self.getDataTill2020()
        else:
            data = self.getDataTill2021()

        (data[metric] / data[metric].shift(1)).plot()
        plt.title(f'Microsoft Quotes Relative Change: {metric}, {year}')
        plt.savefig(f'analysis/reltvchange_{metric}_{year}.png')
        plt.show()

if __name__ == '__main__':
    p = Preprocess(None)
    for y in [2020, 2021]:
        for m in ['Close', 'High', 'Low']:
            p.plotSingleMetric(m, y)
            p.plotIndicators(m, y)
            p.plotAbsoluteChange(m, y)
            p.plotRelativeChange(m, y)