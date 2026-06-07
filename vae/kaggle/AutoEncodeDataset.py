import os
import torch
import torchvision
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from torchvision.transforms import v2


class AutoEncodeDataset(Dataset):
    def __init__(self, annotationFile, imgDir, transform = None):
        self.imgLabels = pd.read_csv(annotationFile)
        self.imgDir = imgDir
        self.transform = transform
        self.resize = v2.Compose([v2.Resize(size = 27), transforms.ToTensor()])

    def __len__(self):
        return len(self.imgLabels)

    def __getitem__(self, idx):
        imgPath = os.path.join(self.imgDir, self.imgLabels.loc[idx, 'file'])
        image = torchvision.io.read_image(imgPath).to(torch.float32) / 255

        if self.transform is not None:
            image = transforms.ToPILImage()(image)
            image = self.transform(image)

        label = transforms.ToPILImage()(image)
        label = self.resize(label)

        return image, label