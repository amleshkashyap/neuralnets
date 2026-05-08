import matplotlib.pyplot as plt
import numpy as np

class Display:
    def plot_latent_space(self, vae, n = 10, figsize = 5):
        img_size = 28
        scale = 0.5
        figure = np.zeros((img_size * n, img_size * n))

        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                sample = np.array([[xi, yi]])
                x_decoded = vae.decoder.predict(sample, verbose = 0)
                images = x_decoded[0].reshape(img_size, img_size)
                figure[
                    i * img_size: (i + 1) * img_size,
                    j * img_size: (j + 1) * img_size
                ] = images

        plt.figure(figsize = (figsize, figsize))
        start_range = img_size // 2
        end_range = n * img_size + start_range
        pixel_range = np.arange(start_range, end_range, img_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)

        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap = "Greys_r")
        plt.savefig('my_plot.png')
        plt.show()