import torch

a = torch.range(0, 11)
a = a.reshape([3, 4])
print(a)
print(a.is_contiguous())

b = torch.range(0, 11)
b = b.reshape([3, 4])
b = b.permute(1, 0)
print(b)
