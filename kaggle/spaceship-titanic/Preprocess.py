import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from Parameters import *

class Preprocess:
    def __init__(self, relativePath, scaler):
        self.filePath = os.getcwd()
        for path in relativePath:
            self.filePath = os.path.join(self.filePath, path)

        self.scaler = scaler
        self.df = None
        self.labels = None
        self.xTrain = None
        self.yTrain = None
        self.xValidate = None
        self.yValidate = None
        self.dataloader = None

    def loadData(self, mode = 'train'):
        self.df = pd.read_csv(self.filePath)
        self.df = self.df.drop("Name", axis = 1)
        self.splitPassengerIds()
        self.splitCabin()
        convertToNumeric = ['HomePlanet', 'CryoSleep', 'CabinDeck', 'CabinSide', 'Destination', 'VIP']
        self.convertToNumber(convertToNumeric)
        fillMissing = convertToNumeric + ['PassengerGroups', 'PassengerNums', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNumber']
        self.fillMissingValues(fillMissing)

        print(self.df.columns)
        if mode == 'test':
            self.df["Transported"] = 0
            self.labels = self.df["Transported"]
            self.df = self.df.drop("Transported", axis = 1)
            self.xTrain = torch.tensor(
                self.scaler.transform(self.df.values.astype(np.float32)),
                dtype = torch.float32
            ).unsqueeze(1)  # Shape: (8693 x 1 x 15)

            self.yTrain = torch.tensor(
                self.labels.values,
                dtype = torch.float32
            )   # Shape: (8693, 1)
        else:
            self.labels = self.df["Transported"]
            self.df = self.df.drop("Transported", axis = 1)
            trainDf, valDf, trainLabel, valLabel = train_test_split(
                self.df,
                self.labels,
                test_size = TRAIN_TEST_SPLIT,
                random_state = 42
            )
            self.xTrain = torch.tensor(
                self.scaler.fit_transform(trainDf.values.astype(np.float32)),
                dtype = torch.float32
            ).unsqueeze(1)

            self.xValidate = torch.tensor(
                self.scaler.transform(valDf.values.astype(np.float32)),
                dtype = torch.float32
            ).unsqueeze(1)

            self.yTrain = torch.tensor(
                trainLabel.values,
                dtype = torch.float32
            ).unsqueeze(1)
            print(self.xTrain.shape, self.yTrain.shape)

            self.yValidate = torch.tensor(
                valLabel.values,
                dtype = torch.float32
            ).unsqueeze(1)

        self.labels = self.labels.values
        # Create DataLoader for mini-batch training
        dataset = TensorDataset(self.xTrain, self.yTrain)
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
        return self.xTrain

    def getXValidate(self):
        return self.xValidate

    def getYTrain(self):
        return self.yTrain

    def getYValidate(self):
        return self.yValidate

    def getLabels(self):
        return self.labels

    def getScaler(self):
        return self.scaler

    def splitPassengerIds(self):
        passengerIds = self.df['PassengerId'].tolist()
        passengerGroup = []
        passengerNum = []

        for passengerId in passengerIds:
            pG, pN = passengerId.split('_')
            passengerGroup.append(int(pG))
            passengerNum.append(int(pN))

        self.df["PassengerGroups"] = pd.Series(passengerGroup).astype('Int64')
        self.df["PassengerNums"] = pd.Series(passengerNum).astype('Int64')
        self.df = self.df.drop("PassengerId", axis = 1)

    def splitCabin(self):
        cabins = self.df['Cabin']
        cabinDeck = []
        cabinNumber = []
        cabinSide = []
        for cabin in cabins:
            if not type(cabin) is str:
                continue
            cD, cN, cS = cabin.split('/')
            cabinDeck.append(cD)
            cabinNumber.append(cN)
            cabinSide.append(cS)

        self.df["CabinDeck"] = pd.Series(cabinDeck).astype('string')
        self.df["CabinNumber"] = pd.Series(cabinNumber).astype('Int64')
        self.df["CabinSide"] = pd.Series(cabinSide).astype('string')

        self.df = self.df.drop("Cabin", axis = 1)

    def convertToNumber(self, convertToNumeric):
        for col in convertToNumeric:
            objects = self.df[col].tolist()
            categories = {}
            count = 1

            for category in objects:
                if categories.get(category) is None:
                    categories[category] = count
                    count += 1

            for i in range(len(objects)):
                if type(objects[i]) is not str:
                    continue
                objects[i] = categories[objects[i]]

            self.df[col] = pd.Series(objects).astype('Int64')
            objects = self.df[[col]]
            objects = objects.fillna(objects.median())
            self.df[col] = objects

    def fillMissingValues(self, fillMissing):
        for col in fillMissing:
            objects = self.df[col].tolist()
            self.df[col] = pd.Series(objects).astype('Int64')
            objects = self.df[[col]]
            objects = objects.fillna(objects.median())
            self.df[col] = objects
