import random
import torch
from torch import optim, nn

from FCNN import FCNN
from InterpolationPredictor import InterpolationPredictor
from DummyPredictor import DummyPredictor
from HwesPredictor import HwesPredictor
from GenerateDataset import GenerateDataset
from Train import Train
from Evaluation import Evaluation

random.seed(1)
torch.manual_seed(1)

if __name__ == "__main__":
    FEATURES = 256

    datasetGenerator = GenerateDataset()
    datasetGenerator.prepareData(FEATURES, 3000)
    XTrain = datasetGenerator.getXTrain()
    XValidate = datasetGenerator.getXValidate()
    XTest = datasetGenerator.getXTest()
    YTrain = datasetGenerator.getYTrain()
    YValidate = datasetGenerator.getYValidate()
    YTest = datasetGenerator.getYTest()

    fcnn = FCNN(FEATURES, 64, 32, 1)
    dummyPredictor = DummyPredictor()
    interpolationPredictor = InterpolationPredictor()
    hwesPredictor = HwesPredictor()

    optimizer = optim.Adam(
        params = fcnn.parameters(),
        lr = 0.005
    )
    criterion = nn.MSELoss()

    trainer = Train(fcnn, optimizer, criterion)
    trainer.train(XTrain, XValidate, YTrain, YValidate, 10000)
    trainer.plotTrainingProgress()

    bestFcnn = trainer.getBestModel()

    evaluator = Evaluation(bestFcnn, dummyPredictor, interpolationPredictor, hwesPredictor, criterion)
    evaluator.compareModels(XTest, YTest)
    evaluator.checkFCNNOnTraining(XTrain, YTrain)
    evaluator.evaluateOnTest(XTest, YTest)
    evaluator.checkStdDev(XTest, YTest)