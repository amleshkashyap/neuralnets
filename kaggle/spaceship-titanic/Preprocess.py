import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from Parameters import *

class Preprocess:
    def __init__(self, relativePath, scaler, categoryData):
        self.filePath = os.getcwd()
        for path in relativePath:
            self.filePath = os.path.join(self.filePath, path)

        self.scaler = scaler
        self.df = None
        self.labels = None
        self.XTrain = []
        self.YTrain = []
        self.XValidate = []
        self.YValidate = []
        self.XTest = []
        self.YTest = []
        self.dataloader = None
        self.categoryData = categoryData

    def loadData(self, mode = 'train'):
        self.df = pd.read_csv(self.filePath)
        self.df = Preprocess.splitPassengerIds(self.df)
        self.df = Preprocess.splitCabin(self.df)
        self.df = Preprocess.splitNameAndAge(self.df)
        convertToNumeric = [
            'HomePlanet',
            'CryoSleep',
            'CabinDeck',
            'CabinSide',
            'Destination',
            'VIP',
            'Name',
            'Surname',
            'Gender'
        ]
        self.df = Preprocess.convertToNumber(self.df, convertToNumeric, self.categoryData)
        otherColumns = [
            'PassengerGroups',
            'PassengerNums',
            'Age',
            'RoomService',
            'FoodCourt',
            'ShoppingMall',
            'Spa',
            'VRDeck',
            'CabinNumber',
            'AgeGroups'
        ]
        fillMissing = convertToNumeric + otherColumns
        self.df = Preprocess.fillMissingValues(self.df, fillMissing)
        self.df.drop(
            'Name',
            axis = 1,
            inplace = True
        )
        self.df.drop(
            'Gender',
            axis = 1,
            inplace = True
        )
        print(self.df.columns)
        if mode == 'test':
            self.df["Transported"] = 0
            # self.df.to_csv('testDataConverted.csv', index=False)
            self.labels = self.df["Transported"]
            self.df = self.df.drop("Transported", axis = 1)
            self.XTrain = torch.tensor(
                self.scaler.transform(self.df.values.astype(np.float32)),
                dtype = torch.float32
            ).unsqueeze(1)  # Shape: (8693 x 1 x 15)

            self.YTrain = torch.tensor(
                self.labels.values,
                dtype = torch.float32
            ).unsqueeze(1)   # Shape: (8693, 1)
        else:
            self.labels = self.df["Transported"]
            # self.df.to_csv('trainDataConverted.csv', index=False)
            self.df = self.df.drop("Transported", axis = 1)
            trainDf, valDf, trainLabel, valLabel = train_test_split(
                self.df,
                self.labels,
                test_size = TRAIN_TEST_SPLIT,
                random_state = 42
            )
            valDf, testDf, valLabel, testLabel = train_test_split(
                valDf,
                valLabel,
                test_size = 0.5,
                random_state = 42
            )

            self.XTrain = torch.tensor(
                self.scaler.fit_transform(trainDf.values.astype(np.float32)),
                dtype = torch.float32
            ).unsqueeze(1)

            self.XValidate = torch.tensor(
                self.scaler.transform(valDf.values.astype(np.float32)),
                dtype = torch.float32
            ).unsqueeze(1)

            self.XTest = torch.tensor(
                self.scaler.transform(testDf.values.astype(np.float32)),
                dtype = torch.float32
            ).unsqueeze(1)

            self.YTrain = torch.tensor(
                trainLabel.values,
                dtype = torch.float32
            ).unsqueeze(1)

            self.YValidate = torch.tensor(
                valLabel.values,
                dtype = torch.float32
            ).unsqueeze(1)

            self.YTest = torch.tensor(
                testLabel.values,
                dtype = torch.float32
            ).unsqueeze(1)

        self.labels = self.labels.values
        # Create DataLoader for mini-batch training
        dataset = TensorDataset(self.XTrain, self.YTrain)
        self.dataloader = DataLoader(
            dataset,
            batch_size = BATCH_SIZE,
            shuffle = True
        )

    def getData(self):
        return self.dataloader

    def getDf(self):
        return self.df

    def getXTrain(self):
        return self.XTrain

    def getXValidate(self):
        return self.XValidate

    def getYTrain(self):
        return self.YTrain

    def getYValidate(self):
        return self.YValidate

    def getXTest(self):
        return self.XTest

    def getYTest(self):
        return self.YTest

    def getLabels(self):
        return self.labels

    def getScaler(self):
        return self.scaler

    def getCategoryData(self):
        return self.categoryData

    @staticmethod
    def splitPassengerIds(df):
        passengerIds = df['PassengerId'].tolist()
        passengerGroup = []
        passengerNum = []

        for passengerId in passengerIds:
            pG, pN = passengerId.split('_')
            passengerGroup.append(int(pG))
            passengerNum.append(int(pN))

        df["PassengerGroups"] = pd.Series(passengerGroup).astype('Int64')
        df["PassengerNums"] = pd.Series(passengerNum).astype('Int64')
        df = df.drop("PassengerId", axis = 1)
        return df

    @staticmethod
    def splitCabin(df):
        cabins = df['Cabin']
        cabinDeck = []
        cabinNumber = []
        cabinSide = []
        for cabin in cabins:
            if not type(cabin) is str:
                cabinDeck.append(np.nan)
                cabinNumber.append(np.nan)
                cabinSide.append(np.nan)
                continue
            cD, cN, cS = cabin.split('/')
            cabinDeck.append(cD)
            cabinNumber.append(cN)
            cabinSide.append(cS)

        df["CabinDeck"] = pd.Series(cabinDeck).astype('string')
        df["CabinNumber"] = pd.Series(cabinNumber).astype('Int64')
        df["CabinSide"] = pd.Series(cabinSide).astype('string')

        df = df.drop("Cabin", axis = 1)
        return df

    @staticmethod
    def splitNameAndAge(df):
        names = df['Name']
        ages = df['Age']

        surnames = []
        for name in names:
            if not type(name) is str:
                surnames.append(np.nan)
                continue
            surname = name.split(" ")[-1]
            surnames.append(surname)

        df["Surname"] = pd.Series(surnames).astype('string')
        ageBuckets = [15, 25, 35, 45, 55, 65, 75, 85, 95]
        ageGroups = []
        for age in ages:
            if not type(age) in [int, float, str]:
                ageBuckets.append(np.nan)
                continue
            count = 0
            for a in ageBuckets:
                if age <= a:
                    ageGroups.append(count)
                    break
                count += 1
        df["AgeGroups"] = pd.Series(ageGroups).astype('Int64')
        return df

    @staticmethod
    def convertToNumber(df, convertToNumeric, categoryData):
        for col in convertToNumeric:
            objects = df[col].tolist()
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

            df[col] = pd.Series(objects).astype('Int64')
            objects = df[[col]]
            objects = objects.fillna(objects.median())
            df[col] = objects

        return df

    @staticmethod
    def fillMissingValues(df, fillMissing):
        for col in fillMissing:
            objects = df[col].tolist()
            df[col] = pd.Series(objects).astype('Int64')
            objects = df[[col]]
            objects = objects.fillna(objects.median())
            df[col] = objects
        return df