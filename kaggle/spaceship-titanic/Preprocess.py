import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os

class Preprocess:
    def __init__(self, relativePath):
        self.filePath = os.getcwd()
        for path in relativePath:
            self.filePath = os.path.join(self.filePath, path)

        self.df = None
        self.dataloader = None
        self.labels = None

    def loadData(self):
        self.df = pd.read_csv(self.filePath)
        self.df = self.df.drop("Name", axis = 1)
        self.splitPassengerIds()
        self.splitCabin()
        convertToNumeric = ['HomePlanet', 'CryoSleep', 'CabinDeck', 'CabinSide', 'Destination', 'VIP']
        self.convertToNumber(convertToNumeric)
        fillMissing = convertToNumeric + ['PassengerGroups', 'PassengerNums', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNumber']
        self.fillMissingValues(fillMissing)
        print(self.df.head())

        yFeatures = self.df[["Transported"]].values
        self.labels = yFeatures
        self.df.drop(
            "Transported",
            axis = 1,
            inplace = True
        )

        print(self.df.columns)
        xFeatures = self.df.values.astype(np.float32)  # Shape: (8693 x 15)
        print(xFeatures)
        xTensor = torch.tensor(
            xFeatures,
            dtype = torch.float32
        ).unsqueeze(1)  # Shape: (8693 x 1 x 15)

        yTensor = torch.tensor(
            yFeatures,
            dtype = torch.float32
        )  # Shape: (8693, 1)

        # Create DataLoader for mini-batch training
        dataset = TensorDataset(xTensor, yTensor)
        self.dataloader = DataLoader(
            dataset,
            batch_size = 16,
            shuffle = True
        )

    def getData(self):
        return self.dataloader

    def getDf(self):
        return self.df

    def getLabels(self):
        return self.labels

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
