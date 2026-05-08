import tensorflow as tf

class ConvolutionNN:
    def __init__(self, image):
        self.image = image
        # every 3x3 block of the image will be multiplied with this kernel to generate a single number
        #   final matrix generated from all block x kernel is called a feature map
        self.kernel = tf.constant([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype = tf.float32)

        self.kernel = tf.reshape(self.kernel, [3, 3, 1, 1])

        self.convolution_output = tf.nn.conv2d(
            input = self.image,
            filters = self.kernel,
            strides = [1, 1, 1, 1],
            padding = 'SAME'
        )

        self.relu_output = None
        self.max_pooling_output = None
        self.flatten_output = None
        self.dense_layer_output = None

    def get_convolution_output(self):
        return self.convolution_output

    def get_relu_output(self):
        if self.relu_output is None:
            self.relu_output = tf.nn.relu(self.convolution_output)

        return self.relu_output

    def get_max_pooling_output(self):
        if self.max_pooling_output is None:
            self.max_pooling_output = tf.nn.max_pool2d(
                input = self.get_relu_output(),
                ksize = [1, 2, 2, 1],
                strides = [1, 2, 2, 1],
                padding = 'SAME'
            )

        return self.max_pooling_output

    def get_flatten_output(self):
        if self.flatten_output is None:
            flatten_layer = tf.keras.layers.Flatten()
            self.flatten_output = flatten_layer(self.get_max_pooling_output())

        print("After Flatten Shape: ", self.flatten_output.shape)
        print("First 20 Flattened Values: ")
        print(self.flatten_output.numpy()[0][:20])

        return self.flatten_output

    def add_dense_layer(self):
        if self.dense_layer_output is None:
            dense_layer = tf.keras.layers.Dense(
                units = 64,
                activation = 'relu'
            )
            self.dense_layer_output = dense_layer(self.get_flatten_output())

        print("After Fully Connected Layer Shape: ", self.dense_layer_output.shape)
        return self.dense_layer_output