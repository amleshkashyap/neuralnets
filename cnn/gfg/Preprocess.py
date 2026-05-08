import tensorflow as tf

class Preprocess:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = tf.io.read_file(self.image_path)
        self.image = tf.io.decode_jpeg(self.image, channels = 1)
        self.image = tf.image.resize(self.image, [300, 300])
        self.image = tf.image.convert_image_dtype(self.image, tf.float32)

    def get_image(self):
        return self.image

    def add_batch_dimension(self):
        self.image = tf.expand_dims(self.image, axis = 0)