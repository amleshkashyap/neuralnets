import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import polars as pl
from datetime import datetime

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
        print(trainDf['date'][:30])
        trainStoreNumbers = trainDf["date"].unique().to_list()
        testDf = pl.read_csv(f'{self.basePath}/testMerged.csv')
        testDf = testDf.sort('date')
        testStoreNumbers = testDf["date"].unique().to_list()
        # trainStoreNumbers.sort(
        #     key = lambda x: datetime.strptime(x, "%Y-%m-%d")
        # )
        # testStoreNumbers.sort(
        #     key = lambda x: datetime.strptime(x, "%Y-%m-%d")
        # )
        print(trainStoreNumbers)
        print(testStoreNumbers)

if __name__ == "__main__":
    p = MergeData()
    # p.mergeData()
    p.splitByStoreNbr()