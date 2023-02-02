import numpy as np
  
# define binary activation Function
def binary(v):
    if v >= 0:
        return 1
    else:
        return -1
  
# design Perceptron Model
def perceptronModel(x, w, b):
    v = np.dot(w, x) + b
    y = binary(v)
    return y
  
# AND Logic Function
# w1 = 1, w2 = 1, b = -1.5
def AND_logicFunction(x):
    w = np.array([1, 1])
    b = -1.5
    return perceptronModel(x, w, b)
  
# testing the Perceptron Model
test1 = np.array([-1, -1])
test2 = np.array([-1, 1])
test3 = np.array([1, -1])
test4 = np.array([1, 1])
  
print("AND({}, {}) = {}".format(-1, -1, AND_logicFunction(test1)))
print("AND({}, {}) = {}".format(-1, 1, AND_logicFunction(test2)))
print("AND({}, {}) = {}".format(1, -1, AND_logicFunction(test3)))
print("AND({}, {}) = {}".format(1, 1, AND_logicFunction(test4)))