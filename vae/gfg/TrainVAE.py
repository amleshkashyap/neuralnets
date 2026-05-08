import keras
import numpy as np

class TrainVAE:
    def __init__(self, vae):
        (x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()
        self.fashion_mnist = np.concatenate([x_train, x_test], axis = 0)
        self.fashion_mnist = np.expand_dims(self.fashion_mnist, -1).astype('float32') / 255
        self.vae = vae

    def train(self):
        self.vae.compile(optimizer = keras.optimizers.Adam())
        self.vae.fit(self.fashion_mnist, epochs = 3, batch_size = 128)