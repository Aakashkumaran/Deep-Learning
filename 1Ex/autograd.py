import torch
from torch.autograd import Variable
 #torch.tensor-->multidimensinal matrix handling
 #requires_grad=True-->tells the autograd for automatic differentiation

a=torch.tensor(5.0,requires_grad=True)

b=torch.tensor(3.0,requires_grad=True)
y=(a**3)/2*b+3
y.backward()
print(a.grad.data)
z=(b**4)*(2*a+b)
z.backward()
print(b.grad.data)