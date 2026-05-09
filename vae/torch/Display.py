import matplotlib.pyplot as plt

class Display:

    def show_images(self, images, labels):
        '''
        :param images: first column should contain image indices, second column should contain flattened image
                       pixels reshaped into 28x28 arrays
        :param labels:
        :return: None
        '''

        pixels = images.reshape(-1, 28, 28)

        # create a figure with subplots for each image
        fig, axs = plt.subplots(
            ncols = len(images), nrows = 1, figsize = (10, 3 * len(images))
        )

        # loop over images and display along with the labels
        for i in range(len(images)):
            # image + label
            axs[i].imshow(pixels[i], cmap = 'gray')
            axs[i].set_title("Label: {}".format(labels[i]))

            # remove tick marks and axis labels
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_xlabel("Index: {}".format(i))

        fig.subplots_adjust(hspace = 0.5)

        plt.savefig('raw_images.png')