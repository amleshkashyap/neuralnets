from Encoder import Encoder
from Decoder import Decoder
from VAEncoder import VAEncoder
from TrainVAE import TrainVAE
from Sampling import Sampling
from Display import Display
import keras
import numpy as np

if __name__ == "__main__":
    sampling = Sampling()
    encoder = Encoder(sampling).get_encoder()
    decoder = Decoder().get_decoder()
    vae = VAEncoder(encoder, decoder)

    trainVae = TrainVAE(vae)
    # trainVae.train()

    display = Display()
    display.plot_latent_space(vae)

    (x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    display.plot_label_clusters(encoder, decoder, x_train, y_train)