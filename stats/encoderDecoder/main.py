import numpy as np
import random
import torch
from EncoderDecoder import EncoderDecoder
from GenerateTimeseries import GenerateTimeseries
from stats.encoderDecoder.Evaluation import Evaluation

seed = 1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

if __name__ == "__main__":
    HIDDEN_SIZE = 64
    TEST_DS_LENGTH = 200
    EPOCHS = 500
    TS_LENGTH = 2000
    TS_HISTORY_LENGTH = 240
    TS_TARGET_LENGTH = 60

    gt = GenerateTimeseries()
    gt.prepareData(
        TS_LENGTH,
        TS_HISTORY_LENGTH,
        TS_TARGET_LENGTH,
        TEST_DS_LENGTH
    )
    XTrain = gt.getXTrain()
    XTest = gt.getXTest()
    YTrain = gt.getYTrain()
    YTest = gt.getYTest()

    model = EncoderDecoder(
        1,
        HIDDEN_SIZE,
        1
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = 0.005
    )

    model.train()
    model.trainModel(
        XTrain,
        YTrain,
        EPOCHS,
        TS_TARGET_LENGTH,
        criterion,
        optimizer,
        method = 'mixedTeacherForcing',
        tfr = 0.05,
    )

    evaluate = Evaluation(model)
    evaluate.evaluate(XTest, YTest, TS_TARGET_LENGTH, TEST_DS_LENGTH)