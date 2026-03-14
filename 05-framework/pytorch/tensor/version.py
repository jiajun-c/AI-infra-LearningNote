import torch

a = torch.zeros([3, 4])
print(a._version)
a.abs_()
print(a._version)
print(a.data_ptr)
a = a + 1
print(a.data_ptr)

print(a._version)