import matplotlib.pyplot as plt
import tensorflow as tf

class Display:
    def display_image(self, image, title, figsize = 10):
        plt.figure(figsize = (figsize, figsize))
        plt.imshow(tf.squeeze(image))
        plt.title(title)
        plt.axis('off')
        plt.savefig('outputs/' + title + '_cnn.png')

