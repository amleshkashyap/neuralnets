from Encoder import Encoder
from Decoder import Decoder
from VAEncoder import VAEncoder
from TrainVAE import TrainVAE
from Display import Display

if __name__ == "__main__":
    encoder = Encoder().get_encoder()
    decoder = Decoder().get_decoder()
    vae = VAEncoder(encoder, decoder)

    trainVae = TrainVAE(vae)
    # trainVae.train()

    display = Display()
    display.plot_latent_space(vae)