import torch
import torch.nn as nn

class BasicExpert(nn.Module):
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.linear = nn.Linear(feature_in, feature_out)
    
    def forward(self, x):
        return self.linear(x)
        # (batch, feature_out)
        
class BasicMOE(nn.Module):
    def __init__(self, feature_in, feature_out, expert_number):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                BasicExpert(feature_in, feature_out) for _ in range(expert_number)
            ]
        )
        self.gate = nn.Linear(feature_in, expert_number)

    def forward(self, x):
        expert_weight = self.gate(x) # (batch, expert_number)
        expert_out_list = [
            expert(x).unsqueeze(1) for expert in self.experts
        ]
        
        # (batch, expert_num, feature_out)
        expert_output = torch.cat(expert_out_list, dim=1)
        
        # (batch, 1, expert_number)
        expert_weight = expert_weight.unsqueeze(1)
        output = expert_weight @ expert_output
        # (batch, 1, feature_out)
        return output.squeeze()

def test_basic_moe():
    # x (batch, feature_in)
    x = torch.rand(2, 4)

    basic_moe = BasicMOE(4, 3, 2)
    out = basic_moe(x)
    print(out)
    print(out.shape)


test_basic_moe()