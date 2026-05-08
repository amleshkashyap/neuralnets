import tensorflow as tf
from keras import layers

class Sampling(layers.Layer):
    '''
    Uses mean to sample z, vector encoding a digit
    '''

    def call(self, inputs):
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon