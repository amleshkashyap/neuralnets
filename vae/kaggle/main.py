import os
import pandas as pd
import numpy as np
import torch
import shutil

from torchvision import transforms
from torchvision.transforms import v2
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch import nn
from warnings import filterwarnings

from vae.kaggle.Display import Display
from vae.kaggle.AutoEncodeDataset import AutoEncodeDataset
from vae.kaggle.Inference import Inference
from vae.kaggle.PreProcess import PreProcess
from vae.kaggle.Train import Train
from vae.kaggle.VariationalAutoEncoder import VariationalAutoEncoder

filterwarnings(action = 'ignore')

if __name__ == '__main__':
    if not os.path.isdir('./train'):
        os.mkdir('./train')

    if not os.path.isdir('./test'):
        os.mkdir('./test')

    PreProcess.doPreprocess('./input/train.csv', './train')

    # prepare test data
    imgFiles = os.listdir('./train')
    # choose any 256 images to move from ./train to ./test directory
    testSample = np.random.choice(imgFiles, size = 256, replace = False)

    for file in tqdm(testSample):
        curr = os.path.join('./train', file)
        new = os.path.join('./test', file)
        shutil.move(curr, new)

    # label.csv has 42000 data points, the full set
    labels = pd.read_csv('./label.csv')
    testSet = labels[labels['file'].isin(testSample)]
    # after this, label.csv has 41744 data points, rows for the 256 images in ./test directory are removed
    labels.drop(testSet.index, axis = 0).to_csv('./label.csv', index = False)

    # transform the training data randomly to have better learning, eg, model should be able to identify even if the numbers appear
    # slightly rotated or distorted in the test/validation set
    trainTransform = v2.Compose([
        v2.RandomAffine(
            degrees = 5,                # may apply a rotation of -5 to 5 degrees
            translate = (0.05, 0.05),   # may translate by 5% both horizontally and vertically
            scale = (0.98, 1.02)        # may scale between 98% and 102%
        ),
        v2.RandomPerspective(
            distortion_scale = 0.5,     # if perspective transforming, then degree of distortion
            p = 0.5
        ),
        transforms.ToTensor()
    ])

    trainingData = AutoEncodeDataset('./label.csv', './train', transform = trainTransform)
    trainSize = int(len(trainingData) * 0.8)
    testSize = len(trainingData) - trainSize

    # split the data as train-test with a ratio of 80:20, randomly
    trainDataset, testDataset = random_split(trainingData, (trainSize, testSize))

    # get an iterable on the train and test datasets with a batch size of 32
    trainDataloader = DataLoader(trainDataset, batch_size = 32, shuffle = True)
    testDataloader = DataLoader(testDataset, batch_size = 32, shuffle = True)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    print(device)

    model = VariationalAutoEncoder(1, 20, 3)

    Display.displayModelProperties(model)

    # mean squared error where the sum of all element-wise errors is taken
    lossFunction = nn.MSELoss(reduction = 'sum')

    images, labels = next(iter(trainDataloader))

    Display.displayLabels(images, labels)
    print(labels[0].shape)

    # AdamW is a variant of Adam that decouples weight decay from gradient updates, apply L2 regularization directly on weights
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)

    # dynamically reduces the learning rate hyperparameter if the evaluation metric (MSELoss in this case) stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1)

    results = Train.trainLoop(
        model,
        lossFunction,
        trainDataloader,
        valLoader = testDataloader,
        optimizer = optimizer,
        lrSchedule = scheduler,
        device = device,
        epochs = 100,
    )

    Display.displayLoss(results)

    inference = Inference(model)
    inputDigit, outs, reconImg = inference.infereence(dataset = testDataset, nExamples = 10)

    Display.displayInferenceResults(inputDigit, outs, reconImg)

    shutil.rmtree("./train/")
    shutil.rmtree("./test/")

    PATH = 'conv-var-autoencoder.pth'
    torch.save(model.state_dict(), PATH)