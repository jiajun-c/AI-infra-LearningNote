import torch
from torch import nn
import torch.nn.functional as F
import math

class ScaledDotProductAttetion(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        # 将Q，K，V放入到一个中
        self.proj = nn.Linear(dim, dim*3)
        self.att_drop = nn.Dropout(0.1)
        self.output_proj = nn.Linear(dim, dim)
    
    def forward(self, X, attention_mask=None):
        QKV = self.proj(X)
        # X shape (batch, seq, dim)
        Q, K, V = torch.split(QKV, self.dim, dim=-1)
        # Q, K, V shape (batch, seq, dim)
        # Q = (batch, seq, dim)
        # k -> (batch, dim, seq)
        att_weight = Q @ K.transpose(-1, -2)/math.sqrt(self.dim)
        # att_weight -> (batch, seq, seq)
        if attention_mask is not None:
            att_weight = att_weight.masked_fill(attention_mask == 0, float('-inf'))
        att_weight = torch.softmax(att_weight, dim=-1)

        att_weight = self.att_drop(att_weight)

        output = att_weight @ V
        # output -> (batch, seq, dim)
        ret = self.output_proj(output)
        return ret
    
X = torch.rand(3, 4, 2)
b = torch.tensor(
    [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ]
)
mask = b.unsqueeze(dim=1).repeat(1, 4, 1)

print(b.shape)

net = ScaledDotProductAttetion(2)
print(net(X, mask).shape)

