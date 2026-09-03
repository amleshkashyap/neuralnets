import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import polars as pl
from datetime import datetime
from pathlib import Path

from Visualize import Visualize


class MergeData:
    def __init__(self):
        self.basePath = os.getcwd()
        self.data = None
        self.holidays = None
        self.oil = None
        self.store = None
        self.transactions = None

    def mergeData(self):
        self.holidays = pd.read_csv(f'{self.basePath}/data/holidays_events.csv')
        self.oil = pd.read_csv(f'{self.basePath}/data/oil.csv')
        self.stores = pd.read_csv(f'{self.basePath}/data/stores.csv')
        self.transactions = pd.read_csv(f'{self.basePath}/data/transactions.csv')
        extraColumns = [
            'oil_price',
            'holiday_type', # normal day, holiday, transferred day
            'holiday_description',
            'city',
            'state',
            'store_type',
            'cluster',
            'transactions'
        ]
        trainDf = pl.read_csv(f'{self.basePath}/data/train.csv')
        for col in extraColumns:
            trainDf.with_columns(
                col = pl.lit(np.nan)
            )

        rows = trainDf.to_dicts()

        with ProcessPoolExecutor() as executor:
            processedRows = list(executor.map(
                self.mergeWithData,
                rows,
                chunksize = 1000
            ))
        print("Completed For Training Data")
        finalTrainDf = pl.from_dicts(processedRows)
        finalTrainDf.write_csv('trainMerged.csv')

        testDf = pl.read_csv(f'{self.basePath}/data/test.csv')
        for col in extraColumns:
            testDf.with_columns(
                col = pl.lit(np.nan)
            )

        rows = testDf.to_dicts()

        with ProcessPoolExecutor() as executor:
            processedRows = list(executor.map(
                self.mergeWithData,
                rows,
                chunksize = 1000
            ))
        finalTestDf = pl.from_dicts(processedRows)
        finalTestDf.write_csv('testMerged.csv')
        print("Completed For Testing Data")

    def mergeWithData(self, row):
        dt = row['date']
        storeNbr = row['store_nbr']
        h = self.holidays.loc[self.holidays['date'] == dt]
        o = self.oil.loc[self.oil['date'] == dt]
        sto = self.stores.loc[self.stores['store_nbr'] == storeNbr]
        txn = self.transactions.loc[(self.transactions['date'] == dt) & (self.transactions['store_nbr'] == storeNbr)]
        if h.empty:
            htype = "Work Day"
            transferred = False
            locale = "National"
            localeName = "Ecuador"
            description = ""
        else:
            htype = "Work Day" if h.type.empty else h.type.iloc[0]
            transferred = False if h.transferred.empty else h.transferred.iloc[0]
            locale = "National" if h.locale.empty else h.locale.iloc[0]
            localeName = "" if h.locale_name.empty else h.locale_name.iloc[0]
            description = "" if h.description.empty else h.description.iloc[0]

        if sto.empty:
            stype = None
            state = None
            city = None
            cluster = None
        else:
            stype = sto['type'].iloc[0]
            state = sto['state'].iloc[0]
            city = sto['city'].iloc[0]
            cluster = sto['cluster'].iloc[0]

        row['holiday_description'] = description
        row['city'] = city
        row['state'] = state
        row['cluster'] = cluster
        row['store_type'] = stype
        row['oil_price'] = np.nan if (o.empty or o['dcoilwtico'].empty) else o['dcoilwtico'].iloc[0]
        row['transactions'] = np.nan if (txn.empty or txn['transactions'].empty) else txn['transactions'].iloc[0]

        if locale == "National":
            if transferred == True:
                row['holiday_type'] = "Transferred Work Day"
            else:
                row['holiday_type'] = htype
        elif (state == localeName or city == localeName):
            if transferred:
                row['holiday_type'] = "Transferred Work Day"
            else:
                row['holiday_type'] = htype
        else:
            row['holiday_type'] = "Work Day"

        return row

    def splitByStoreNbr(self):
        trainDf = pl.read_csv(f'{self.basePath}/trainMerged.csv')
        trainDf = trainDf.sort('date')
        trainStoreNumbers = trainDf["store_nbr"].unique().to_list()
        for storeNbr in trainStoreNumbers:
            values = self.fetchByColumn(trainDf, 'store_nbr', storeNbr)
            self.writeToCsv(values, f'data/train/{storeNbr}.csv')

        testDf = pl.read_csv(f'{self.basePath}/testMerged.csv')
        testDf = testDf.sort('date')
        testStoreNumbers = testDf["store_nbr"].unique().to_list()
        for storeNbr in testStoreNumbers:
            values = self.fetchByColumn(testDf, 'store_nbr', storeNbr)
            self.writeToCsv(values, f'data/test/{storeNbr}.csv')

    def fetchByColumn(self, df: pl.DataFrame, col, value):
        return df.filter(pl.col(col) == value)

    def writeToCsv(self, df: pl.DataFrame, filePath):
        df.write_csv(filePath)

    def generateTrainMetrics(self):
        pass


if __name__ == "__main__":
    p = MergeData()
    # p.mergeData()
    # p.splitByStoreNbr()
    basePath = os.getcwd()
    trainPath = Path(f'{basePath}/../data/train/')
    testPath = Path(f'{basePath}/../data/test')

    vis = Visualize()

    families = []
    for file in trainPath.glob("*.csv"):
        df = pl.read_csv(file)
        cols = df.columns
        print(file.name, ": ", df.height)
        for col in cols:
            # if col in ['id', 'store_nbr']:
            #     continue
            if col not in ['family']:
                continue
            f = df[col].unique().to_list()
            for i in f:
                if i in families:
                    continue
                families.append(i)
            values = df.select(pl.col(col).value_counts(sort = True))
            # print(file.name, ": ", col, values)
        # vis.plotBasicHistogram(df["sales"].to_list(), 30)
        # vis.plotBasicHistogram(df["family"].to_list(), 30)
    print(len(families))
    print(families)

    for file in testPath.glob("*.csv"):
        df = pl.read_csv(file)
        cols = df.columns
        print(file.name, ": ", df.height)
        for col in cols:
            # if col in ['id', 'store_nbr']:
            #     continue
            if col not in ['date', 'family']:
                continue
            f = df[col].unique().to_list()
            for i in f:
                if i in families:
                    continue
                families.append(i)
            values = df.select(pl.col(col).value_counts(sort = True))
            print(file.name, ": ", col, values)
        # vis.plotBasicHistogram(df["sales"].to_list(), 30)
        # vis.plotBasicHistogram(df["family"].to_list(), 30)
    print(len(families))
    print(families)