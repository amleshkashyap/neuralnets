from Display import Display
from Preprocess import Preprocess
from ConvolutionNN import ConvolutionNN


if __name__ == "__main__":
    preprocess = Preprocess('sample_image.png')
    image = preprocess.get_image()
    display = Display()

    display.display_image(image, "Original_Image")

    preprocess.add_batch_dimension()
    image = preprocess.get_image()

    cnn = ConvolutionNN(image)

    display.display_image(cnn.get_convolution_output(), "After_CNN")

    display.display_image(cnn.get_relu_output(), "After_ReLU_Activation")

    display.display_image(cnn.get_max_pooling_output(), "After_Max_Pooling")

    cnn.get_flatten_output()
    cnn.add_dense_layer()