import urllib.request as request
import gzip
import numpy as np

class Preprocess:
    def get_mnist_data(self):
        url = "http://yann.lecun.com/exdb/mnist/"
        filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
        data = []
        for filename in filenames:
            print("Downloading " + filename)
            request.urlretrieve(url + filename, filename)
            with gzip.open(filename, 'rb') as f:
                if 'labels' in filename:
                    # load labels as 1D array
                    data.append(np.frombuffer(f.read(), np.uint8, offset=8))
                else:
                    # load images as 2D array of pixels
                    data.append(np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28))

        X_train, X_test, y_train, y_test = data

        # normalize
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

        # convert labels to integers
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)