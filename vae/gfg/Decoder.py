import keras
from keras import layers

class Decoder:
    latent_dim = 2
    latent_inputs = keras.Input(shape = (latent_dim, ))
    decoder = None

    def __init__(self):
        x = layers.Dense(7 * 7 * 64, activation = 'relu')(self.latent_inputs)

        x = layers.Reshape((7, 7, 64))(x)

        x = layers.Conv2DTranspose(128,
                                   3,
                                   activation = 'relu',
                                   strides = 2,
                                   padding = 'same')(x)

        x = layers.Conv2DTranspose(64,
                                   3,
                                   activation = 'relu',
                                   strides = 2,
                                   padding = 'same')(x)

        decoder_outputs = layers.Conv2DTranspose(1,
                                                 3,
                                                 activation = 'sigmoid',
                                                 padding = 'same')(x)

        self.decoder = keras.Model(self.latent_inputs, decoder_outputs, name = "decoder")

    def decoder_summary(self):
        return self.decoder.summary()

    def get_decoder(self):
        return self.decoder