import numpy as np
import torch

# define binary activation Function
def binary(v):
    if v >= 0:
        return 1
    else:
        return -1

# design Perceptron Model
def perceptronModel(x, w, b):
    v = torch.tensor(torch.dot(w, x) + b)
    y = binary(v)
    return y

# AND Logic Function
# w1 = 1, w2 = 1, b = -1.5
def AND_logicFunction(x):
    w = torch.tensor(np.array([1, 1]))
    b = -1.5
    return perceptronModel(x, w, b)

# testing the Perceptron Model
test1 = torch.tensor(np.array([-1, -1]))
test2 = torch.tensor(np.array([-1, 1]))
test3 = torch.tensor(np.array([1, -1]))
test4 = torch.tensor(np.array([1, 1]))

print("AND({}, {}) = {}".format(-1, -1, AND_logicFunction(test1)))
print("AND({}, {}) = {}".format(-1, 1, AND_logicFunction(test2)))
print("AND({}, {}) = {}".format(1, -1, AND_logicFunction(test3)))
print("AND({}, {}) = {}".format(1, 1, AND_logicFunction(test4)))