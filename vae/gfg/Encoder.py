import keras
from keras import layers
from Sampling import Sampling

class Encoder:
    latent_dim = 2
    encoder_inputs = keras.Input(shape=(28, 28, 1))
    encoder = None
    z = None
    mean = None
    log_var = None

    def __init__(self):
        x = layers.Conv2D(64,
                          3,
                          activation = 'relu',
                          padding = 'same')(self.encoder_inputs)

        x = layers.Conv2D(128,
                          3,
                          activation = 'relu',
                          strides = 2,
                          padding = 'same')(x)

        x = layers.Flatten()(x)

        x = layers.Dense(16,
                          activation = 'relu')(x)

        self.mean = layers.Dense(self.latent_dim, name = "mean")(x)
        self.log_var = layers.Dense(self.latent_dim, name = "log_var")(x)

        self.z = Sampling()([self.mean, self.log_var])

        self.encoder = keras.Model(self.encoder_inputs, [self.mean, self.log_var, self.z], name = "encoder")

    def encoder_summary(self):
        return self.encoder.summary()

    def get_latent_dim(self):
        return self.latent_dim

    def get_encoder(self):
        return self.encoder


