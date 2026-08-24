from RecurrentNN import RecurrentNN
from GRU import GRU
from LSTM import LSTM
from Preprocess import Preprocess
from Evaluation import Evaluation
from EvaluationLSTM import EvaluationLSTM
from Train import Train
from sklearn.preprocessing import MinMaxScaler

def runAny(preprocess, className, name, data):
    scaler = MinMaxScaler()
    FEATURES = 240
    testLen = 300
    HIDDEN_SIZE = 24
    EPOCHS = 500
    scaledData = scaler.fit_transform(data)
    preprocess.prepareTrainingData(scaledData, FEATURES, testLen)
    XTrain = preprocess.getXTrain()
    XValidate = preprocess.getXValidate()
    XTest = preprocess.getXTest()
    YTrain = preprocess.getYTrain()
    YValidate = preprocess.getYValidate()
    YTest = preprocess.getYTest()
    model = className(
        1,
        HIDDEN_SIZE,
        1
    )
    trainer = Train(
        model,
        name
    )
    trainer.train(XTrain, XValidate, YTrain, YValidate, EPOCHS)
    bestModel = trainer.loadBestModel()
    if name == 'lstm':
        evaluater = EvaluationLSTM(
            bestModel,
            trainer.getCriterion(),
            scaler
        )
    else:
        evaluater = Evaluation(
            bestModel,
            trainer.getCriterion(),
            name,
            scaler
        )

    trainer.plotTrainingProgress()
    evaluater.evaluate(XValidate, XTest)
    evaluater.plotEvaluation(YTest)

if __name__ == "__main__":
    preprocess = Preprocess()
    data = preprocess.getAepTimeseries()
    preprocess.viewAepTimeseries()
    runAny(preprocess, RecurrentNN, "rnn", data)
    data = preprocess.getPjmeTimeseries()
    preprocess.viewPjmeTimeseries()
    runAny(preprocess, GRU, "gru", data)
    data = preprocess.getNiTimeseries()
    preprocess.viewNiTimeseries()
    runAny(preprocess, LSTM, "lstm", data)