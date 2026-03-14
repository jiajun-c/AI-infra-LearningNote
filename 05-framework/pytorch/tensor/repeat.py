import torch

x = torch.tensor([[1, 2], 
                  [3, 4]]) # Shape: (2, 2)
t4 = torch.tensor([1.9, 2.5], dtype=torch.int32)
print(x.shape)
# 在 dim=0 (行) 上每行重复 2 次
out = torch.repeat_interleave(x, repeats=2, dim=0)
print(out.shape)
print(out)
# tensor([[1, 2],
#         [1, 2],  <-- Row 0 repeated
#         [3, 4],
#         [3, 4]]) <-- Row 1 repeated

y = torch.arange(0, 3)
outy = torch.repeat_interleave(y, repeats=torch.tensor([1, 2, 3], dtype=torch.long))
print(outy)
