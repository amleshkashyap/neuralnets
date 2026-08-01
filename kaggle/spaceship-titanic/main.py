from RecurrentNN import RecurrentNN
from Train import Train
from Preprocess import Preprocess
import torch.nn as nn
import torch.optim as optim

from Inference import Inference

if __name__ == "__main__":
    trainFilePath = ["data", "train.csv"]
    testFilePath = ["data", "train.csv"]

    preprocess = Preprocess(trainFilePath)
    preprocess.loadData()
    dataloader = preprocess.getData()

    INPUT_SIZE = 15
    HIDDEN_SIZE = 20
    OUTPUT_SIZE = 1

    model = RecurrentNN(
        INPUT_SIZE,
        HIDDEN_SIZE,
        OUTPUT_SIZE
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr = 0.005
    )
    trainer = Train(
        model,
        dataloader,
        criterion,
        optimizer
    )

    trainer.train()

    preprocessTest = Preprocess(testFilePath)
    preprocessTest.loadData()
    testDf = preprocessTest.getDf()

    inference = Inference(model, criterion)
    inference.predict(testDf, preprocessTest.getLabels())