from sklearn.preprocessing import StandardScaler
from Preprocess import Preprocess
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# input features
INPUT_SIZE = 17

# binary classification has a single valued output
OUTPUT_SIZE = 1

if __name__ == "__main__":
    trainFilePath = ["..", "data", "train.csv"]
    testFilePath = ["..", "data", "test.csv"]
    bestResPath = "../resultsBest.csv"

    scaler = StandardScaler()

    # preprocess the training dataset and the DataLoader
    preprocess = Preprocess(trainFilePath, scaler, {})
    preprocess.loadData()
    dataloader = preprocess.getData()
    categoryData = preprocess.getCategoryData()
    XTrain = preprocess.getXTrain().squeeze(1)
    YTrain = preprocess.getYTrain().squeeze(1)

    XValidate = preprocess.getXValidate().squeeze(1)
    YValidate = preprocess.getYValidate().squeeze(1)

    XTest = preprocess.getXTest().squeeze(1)
    YTest = preprocess.getYTest().squeeze(1)

    qda = QuadraticDiscriminantAnalysis(
        reg_param = 0.01
    )

    qda.fit(XTrain, YTrain)

    preprocessTest = Preprocess(testFilePath, preprocess.getScaler(), categoryData)
    preprocessTest.loadData('test')
    XTrain1 = preprocessTest.getXTrain().squeeze(1)
    YTrain1 = preprocessTest.getYTrain().squeeze(1)

    yPred = qda.predict(XTrain1)
    print(yPred[:20])

    np.savetxt(
        "results.csv",
        yPred.astype(bool),
        delimiter=',',
        fmt='%s'
    )