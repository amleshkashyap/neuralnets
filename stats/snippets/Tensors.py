import random
import matplotlib.image as mpimg
import torch
from torch.autograd import grad
from torchviz import make_dot
from matplotlib import pyplot as plt
import os

os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

class Tensors:
    def __init__(self):
        pass

    def basicTensors(self):
        # 1D tensor
        x = torch.tensor(data = [1, 2, 3])
        print("Basic Tensors")
        print(x)

        # 2D tensor
        x = torch.tensor(data = [[1, 2, 3], [4, 5, 6]])
        print(x)

        # with dtype and grad enabled
        x = torch.tensor(
            data = [1.3, 0.5, 3],
            dtype = torch.float32,
            requires_grad = True
        )
        print("\nTensor with dtype and grad enabled")
        print(x)

        # as a recommended practice, set the following two as early as possible for reproducibility
        torch.manual_seed(0)
        random.seed(0)

        x = torch.rand((2, 2, 2))
        print("\nRandom Tensor")
        print(x)

        # other common tensors
        x = torch.zeros((2, 2))
        print("\nZeros")
        print(x)

        x = torch.ones((2, 2))
        print("Ones")
        print(x)

        x = torch.eye(3)
        print("\nIdentity Tensor")
        print(x)

        print("\nReshaping to (2, 2)")
        x = torch.tensor([1, 2, 3, 4])
        print(x)
        y = x.reshape((2, 2))
        print(y)

        print("\nComputing Gradient")
        x = torch.tensor(
            data = [7, 8, 9],
            dtype = torch.float32,
            requires_grad = True
        )
        # f = x^2
        f = x.pow(2)
        # gradient = 2x
        f.backward(gradient = torch.ones_like(f))
        print(x.grad)


    # build a simple function and compute its output using pytorch Tensor
    def getFunctions(self, x1Val = 0, x2Val = 0, x3Val = 0, x4Val = 0):
        x1 = torch.tensor(
            x1Val,
            requires_grad = True,
            dtype = torch.float32
        )

        x2 = torch.tensor(
            x2Val,
            requires_grad = True,
            dtype = torch.float32
        )

        x3 = torch.tensor(
            x1Val,
            requires_grad=True,
            dtype=torch.float32
        )

        x4 = torch.tensor(
            x1Val,
            requires_grad=True,
            dtype=torch.float32
        )

        # function - (x1^3) * x2 + (x3 * x4)
        p1 = x1.pow(3)
        m1 = p1 * x2
        m2 = x3 * x4
        f = m1 + m2

        vars = {
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'x4': x4
        }
        return f, vars

if __name__ == '__main__':
    t = Tensors()

    t.basicTensors()

    f, params = t.getFunctions(2, 4, 3, 5)
    print("\nEvaluated Function")
    print(f.item())

    # print the computational graph for the above function
    make_dot(f, params).render(
        "f_torchviz",
        format = "png",
        cleanup = True
    )
    img = mpimg.imread("f_torchviz.png")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.show()

    # get the partial derivative wrt x1
    dfDx1 = grad(
        outputs = f,
        inputs = [params['x1']]
    )
    print(dfDx1)

    # get the full gradient
    f, params = t.getFunctions(2, 4, 3, 5)
    dfDx = grad(
        outputs = f,
        inputs = params.values()
    )
    print(dfDx)