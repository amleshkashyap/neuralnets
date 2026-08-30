from sklearn.preprocessing import StandardScaler
from CompareTrainTest import CompareTrainTest

if __name__ == "__main__":
    trainFilePath = ["..", "data", "train.csv"]
    testFilePath = ["..", "data", "test.csv"]
    bestResPath = "../resultsBest.csv"

    scaler = StandardScaler()

    compareTrainTest = CompareTrainTest(trainFilePath, testFilePath, scaler)
    compareTrainTest.loadData()
    compareTrainTest.compareTrainTest()