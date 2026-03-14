import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicExpert(nn.Module):
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.linear = nn.Linear(feature_in, feature_out)

    def forward(self, x):
        return self.linear(x)

class SparseMoE(nn.Module):
    def __init__(self, feature_in, feature_out, expert_num, top_k):
        super().__init__()
        self.experts = nn.ModuleList(
            BasicExpert(feature_in, feature_out)
        )
        self.top_k = top_k
        self.gate = nn.Linear(feature_in, expert_num)
    
    def forward(self, x):
        
