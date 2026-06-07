import numpy as np
import csv

class FileUtils:
    @staticmethod
    def readData(filename):
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            reader.__next__()

            for row in reader:
                yield np.array(row, dtype = float)

    @staticmethod
    def moveTo(obj, device):
        if isinstance(obj, list):
            return [FileUtils.moveTo(x, device) for x in obj]

        elif isinstance(obj, tuple):
            return tuple(FileUtils.moveTo(obj, device))

        elif isinstance(obj, set):
            return set(FileUtils.moveTo(obj, device))

        elif isinstance(obj, dict):
            toReturn = dict()
            for key, value in obj.items():
                toReturn[key] = FileUtils.moveTo(value, device)
            return toReturn

        elif hasattr(obj, 'to'):
            return obj.to(device)

        else:
            return obj