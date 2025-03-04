# MoE (Mixed of Experts)

混合专家模型是一种深度学习架构，其将神经网络划分为若干个专家，同时使用门控网络的方式来选择输入样本应该由哪些专家进行计算
其相比于稠密模型，预训练速度更快，与具有相同参数数量的模型相比，推理速度更快，其缺点在于需要大量显存，因为所有的专家系统都
需要被加载到内存中

## 1. 基础的MoE

一个简单的专家模型如下所示，其中只有一个线性层

```python3
class BasicExpert(nn.Module):
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.linear = nn.Linear(feature_in, feature_out)
    
    def forward(self, x):
        return self.linear(x)
        # (batch, feature_out)
```

使用一个`gate`来选择使用的专家，将专家们的输入和权重相乘，得到最后的输出。

```python3
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
```

## 2. 稀疏MoE

在基础的MoE中，其会选择全部的专家进行输出，而在稀疏的MoE中，其会选择topK的输出结果进行加权求和，并把输入的样本变为大模型中真实的输入shape

```python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from basic import *

class MoERouter(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, expert_number)
        self.expert_number = expert_number
        self.top_k = top_k
    
    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        
        router_weights, selected_experts = torch.topk(
            routing_probs,
            self.top_k,
            dim=-1
        )
        # (b*s, topk)
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(hidden_states.dtype)

        expert_mask = F.one_hot(
            selected_experts,
            num_classes=self.expert_number
        )
        # (b*s, topk, expert_number)

        expert_mask = expert_mask.permute(2, 1, 0)
        # (expert_number, topk, b*s)
        return router_logits, router_weights, selected_experts, expert_mask

class MOEConfig:
    def __init__(
        self,
        hidden_dim,
        expert_number,
        top_k,
        shared_experts_number =2,
    ):
        self.hidden_dum = hidden_dim
        self.expert_number = expert_number
        self.top_k = top_k
        self.shared_experts_number = shared_experts_number

class SparseMOE(nn.Module):
    def __init__(self, config:MOEConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dum
        self.expert_number = config.expert_number
        self.top_k = config.top_k
        
        self.experts = nn.ModuleList(
            [
                BasicExpert(self.hidden_dim, self.hidden_dim) for _ in range(self.expert_number)
            ]
        )
        self.router = MoERouter(self.hidden_dim, self.expert_number, self.top_k)
    
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        hidden_states = x.view(-1, hidden_dim)
        
        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(hidden_states)
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states.unsqueeze(
                0
            )[:, top_x, :].reshape(-1, hidden_dim) 
            current_hidden_states = expert_layer(
                current_state
            ) * router_weights[top_x, idx].unsqueeze(-1)  # （selected_token_number, 1） 这里有广播
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            
        return final_hidden_states, router_logits # shape 是 (b * s, expert_number)
def test_token_level_moe():
    x = torch.rand(2, 4, 16)
    config = MOEConfig(16, 2, 2)
    token_level_moe = SparseMOE(config)
    out = token_level_moe(x)
    print(out[0].shape, out[1].shape)


test_token_level_moe()
```