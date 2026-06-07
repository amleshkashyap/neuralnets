import os
import cv2 as cv
import pandas as pd
from vae.kaggle.FileUtils import FileUtils


class PreProcess:
    @staticmethod
    def getLabelAndImage(array):
        return array[1:], array[0]

    @staticmethod
    def convertToImg(array, filename):
        img = array.reshape((28, 28))
        cv.imwrite(filename, img)

    @staticmethod
    def doPreprocess(filename, dstDir):
        Y = []

        for i, arr in enumerate(FileUtils.readData(filename)):
            img, label = PreProcess.getLabelAndImage(arr)
            file = f'img_{i}.jpg'

            Y.append([file, label])
            dst = os.path.join(dstDir, file)
            PreProcess.convertToImg(img, dst)

        df = pd.DataFrame(Y, columns = ['file', 'label'])
        df.to_csv('label.csv', index = False)