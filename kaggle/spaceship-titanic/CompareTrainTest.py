from Preprocess import *
import math

class CompareTrainTest:
    def __init__(self, trainPath, testPath, scaler):
        self.testFilePath = os.getcwd()
        self.trainFilePath = os.getcwd()
        for path in trainPath:
            self.trainFilePath = os.path.join(self.trainFilePath, path)

        for path in testPath:
            self.testFilePath = os.path.join(self.testFilePath, path)

        self.scaler = scaler
        self.df = None
        self.testDf = None
        self.labels = None

    def loadData(self, mode = 'train'):
        self.df = pd.read_csv(self.trainFilePath)
        self.testDf = pd.read_csv(self.testFilePath)
        self.labels = self.df['Transported']
        self.df.drop(
            'Transported',
            axis = 1,
            inplace = True
        )
        categoryData = {}
        self.df = Preprocess.splitPassengerIds(self.df)
        self.testDf = Preprocess.splitPassengerIds(self.testDf)
        self.df = Preprocess.splitCabin(self.df)
        self.testDf = Preprocess.splitCabin(self.testDf)
        self.df = Preprocess.splitNameAndAge(self.df)
        self.testDf = Preprocess.splitNameAndAge(self.testDf)
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
        self.df = Preprocess.convertToNumber(self.df, convertToNumeric, categoryData)
        self.testDf = Preprocess.convertToNumber(self.testDf, convertToNumeric, categoryData)
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
        self.testDf = Preprocess.fillMissingValues(self.testDf, fillMissing)
        self.df.drop(
            'Name',
            axis = 1,
            inplace = True
        )
        self.testDf.drop(
            'Name',
            axis = 1,
            inplace = True
        )
        self.df.drop(
            'Gender',
            axis = 1,
            inplace = True
        )
        self.testDf.drop(
            'Gender',
            axis = 1,
            inplace = True
        )
        print(self.df.columns)
        self.scaler.fit_transform(self.df.values.astype(np.float32))
        self.scaler.transform(self.testDf.values.astype(np.float32))

    def compareTrainTest(self):
        allDistances = []
        labels = []
        for idx, row in self.testDf.iterrows():
            distances = []
            for idx1, row1 in self.df.iterrows():
                distances.append((round(math.dist(row, row1), 2), idx1))
            distances.sort()
            allDistances.append(distances[0][0])
            labels.append(self.labels[distances[0][1]])
        tempDf = pd.DataFrame()
        # tempDf['PassengerId'] = self.testDf['PassengerId']
        tempDf['Transported'] = labels
        tempDf.to_csv(
            "nearestTrain.csv",
            index = False
        )
        print(allDistances)