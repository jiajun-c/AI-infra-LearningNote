import torch

x = torch.tensor([[1, 2], 
                  [3, 4]]) # Shape: (2, 2)
print(x.shape)
# 在 dim=0 (行) 上每行重复 2 次
out = torch.repeat_interleave(x, repeats=2, dim=0)
print(out.shape)
print(out)
# tensor([[1, 2],
#         [1, 2],  <-- Row 0 repeated
#         [3, 4],
#         [3, 4]]) <-- Row 1 repeated