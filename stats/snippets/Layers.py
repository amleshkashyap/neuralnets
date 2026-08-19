import torch
import random
from torch.nn.parameter import Parameter
from matplotlib import pyplot as plt

torch.manual_seed(0)
random.seed(0)

class Layers:
    def __init__(self):
        pass

    def linearLayer(self):
        ll = torch.nn.Linear(3, 2)
        print("Default Weights")
        print(ll.weight)

        # simple computation of the form Ax + b
        x = torch.tensor(data = [1, 2, 3]).float()
        ll = torch.nn.Linear(3, 2)
        ll.weight = torch.nn.Parameter(torch.tensor([[0, 2, 5], [1, 0, 2]]).float())
        ll.bias = torch.nn.Parameter(torch.tensor([1, 1]).float())

        print("\nLinear Layer")
        print(f'x: {x.tolist()}')
        print(f'A: {ll.weight.tolist()}')
        print(f'b: {ll.bias.tolist()}')
        print(f'y = Ax + b: {ll(x).tolist()}')

    def convLayer(self):
        A = torch.tensor([[[
            [1, 2, 0, 1],
            [-1, 0, 3, 2],
            [1, 3, 0, 1],
            [2, -2, 1, 0]
        ]]]).float()
        conv2d = torch.nn.Conv2d(
            1,
            1,
            kernel_size = 2,
            bias = False
        )

        conv2d.weight = Parameter(torch.tensor([[[[1, -1], [-1, 1]]]]).float())
        output = conv2d(A)
        print("\n2D Convolution Layer With Single Output Channel")
        print(output)

    def convLayerMultiOutChannels(self):
        A = torch.tensor([[[1, 0, 2, 0, 3, 0]]]).float()
        conv1d = torch.nn.Conv1d(
            1,
            out_channels = 2,
            kernel_size = 3,
            bias = False
        )
        # here, the 1D vector A is convolved with 2 1D-kernels (given below) - producing 2 outputs
        conv1d.weight = Parameter(torch.tensor([[[1, 0, -1]], [[0, 2, 0]]]).float())
        output = conv1d(A)
        print("\n1D Convolution Layer With 2 Output Channels")
        print(output)

    def padding(self):
        A = torch.tensor([[[1, 0, 2, -1]]]).float()
        conv1d = torch.nn.Conv1d(
            1,
            1,
            kernel_size = 3,
            bias = False,
            padding = 2
        )
        # above will modify the tensor A to [0, 0, 1, 0, 2, -1, 0, 0] - 2 zeros on both sides
        conv1d.weight = Parameter(torch.tensor([[[1, 0, -1]]]).float())
        output = conv1d(A)
        print("\n1D Convolution Layer With Padding = 2")
        print(output)

    def stride(self):
        A = torch.tensor([[[1, 2, 3, 4, 5]]]).float()
        conv1d = torch.nn.Conv1d(
            1,
            1,
            kernel_size = 3,
            bias = False,
            stride = 2
        )
        # here, the kernel will slide 2 cells of the tensor instead of 1
        # ie, move from [1, 2, 3] to [3, 4, 5] - instead of moving to [2, 3, 4]
        conv1d.weight = Parameter(torch.tensor([[[1, 0, -1]]]).float())
        output = conv1d(A)
        print("\n1D Convolution Layer With Stride = 2")
        print(output)

    def maxPoolLayer(self):
        A = torch.tensor([[
            [1, 2, -1, 1],
            [0, 1, -2, -1],
            [3, 0, 5, 0],
            [0, 1, 4, -3]
        ]]).float()
        # here, 2 x 2 submatrices are picked in the given tensor above, and the max value is selected in output
        maxPool = torch.nn.MaxPool2d(2)
        output = maxPool(A)
        print("\nMax Pool Layer")
        print(output.tolist())

    def avgPoolLayer(self):
        A = torch.tensor([[
            [1, 2, -1, 1],
            [0, 1, -2, -1],
            [3, 0, 5, 0],
            [0, 1, 4, -3]
        ]]).float()
        # here, 2 x 2 submatrices are picked in the given tensor above, and their mean is selected in output
        avgPool = torch.nn.AvgPool2d(2)
        out = avgPool(A)
        print("\nAverage Pool Layer")
        print(out.tolist())

    def dropoutLayer(self):
        print("\nDropout Layer")
        x = torch.randint(10, (5, )).float()
        print(f'Initial Tensor: {x}')
        dropout = torch.nn.Dropout(p = 0.5)
        dropout.train()
        r = dropout(x)
        print(f'Dropout In Training Mode: {r}')
        dropout.eval()
        # here, the output r is same as the input x since dropout will be disabled
        r = dropout(x)
        print(f'Dropout In Eval Mode: {r}')

    def losses(self):
        # loss = ((1 - 1) + (5 - 2))/2 = 1.5
        a = torch.tensor([1, 2]).float()
        b = torch.tensor([1, 5]).float()
        absLoss = torch.nn.L1Loss()
        absError = absLoss(a, b)
        print(f'\nL1 Loss ABS: {absError.item()}')

    def relu(self):
        # generates 100 elements in the tensor
        x = torch.linspace(-10, 10, 100)
        print("Input For Activations")
        print(x)
        relu = torch.nn.ReLU()
        y = relu(x)
        plt.title('ReLU')
        plt.plot(x.tolist(), y.tolist())
        plt.show()

    def sigmoid(self):
        x = torch.linspace(-10, 10, 100)
        sigmoid = torch.nn.Sigmoid()
        y = sigmoid(x)
        plt.title('Sigmoid')
        plt.plot(x.tolist(), y.tolist())
        plt.show()

    def tanh(self):
        x = torch.linspace(-10, 10, 100)
        tanh = torch.nn.Tanh()
        y = tanh(x)
        plt.title('Tanh')
        plt.plot(x.tolist(), y.tolist())
        plt.show()

if __name__ == '__main__':
    layer = Layers()
    layer.linearLayer()
    layer.convLayer()
    layer.convLayerMultiOutChannels()
    layer.padding()
    layer.stride()
    layer.maxPoolLayer()
    layer.avgPoolLayer()
    layer.dropoutLayer()
    layer.losses()
    layer.relu()
    layer.sigmoid()
    layer.tanh()