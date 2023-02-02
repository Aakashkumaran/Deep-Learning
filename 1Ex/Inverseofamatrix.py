import torch
a=torch.tensor([[1,2],[3,4]],dtype=torch.int64)
# a=torch.tensor(a,torch.dtype='float32')
print(a)
determinant=a[0][0]*a[1][1]-a[0][1]*a[1][0]
print(determinant)
adj_a=torch.ones(2,2)
t=a[0][0]
adj_a[0][0]=a[1][1]
adj_a[1][1]=t
adj_a[0][1]=-1*a[0][1]
adj_a[1][0]=-1*a[1][0]
print(adj_a)
inverse=(1/determinant)*adj_a
print(inverse)
