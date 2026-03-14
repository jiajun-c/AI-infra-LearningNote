# MoE (Mixed of Experts)

混合专家模型是一种深度学习架构，其将神经网络划分为若干个专家，同时使用门控网络的方式来选择输入样本应该由哪些专家进行计算(token level)
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

在基础的MoE中，其会选择全部的专家进行输出，而在稀疏的MoE中，其会选择topK的输出结果选择专家，在这里其实就有两种设计思路

1. 先确定expert，再确定每个expert要处理的token，然后进行计算
2. 先确定token， 再确定每个token要被哪些expert计算

先确定token的版本如下所示

```python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseMOE(nn.Module):
    def __init__(self, feature_in, feature_out, expert_number, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([
            BasicExpert(feature_in, feature_out) for _ in range(expert_number)
        ])
        self.gate = nn.Linear(feature_in, expert_number)
        self.top_k = top_k

    def forward(self, x):
        # 1. 计算路由分数 (Batch, Expert_Num)
        router_logits = self.gate(x)
        
        # 2. 【核心】只选分数最高的 Top-k 个专家
        # indices: 选中的专家ID, values: 对应的分数
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        
        # 3. 对权重做 Softmax (通常只在选中的 Top-k 里做归一化)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        final_output = torch.zeros(x.shape[0], self.experts[0].linear.out_features)
        
        # 遍历每一个 Batch 里的样本
        for i in range(x.shape[0]):
            # 对于第 i 个样本，只计算它选中的那 k 个专家
            for k in range(self.top_k):
                expert_idx = selected_experts[i, k].item() # 拿到专家ID
                weight = routing_weights[i, k]             # 拿到权重
                
                # 【关键】只运行选中的专家
                expert_out = self.experts[expert_idx](x[i].unsqueeze(0))
                
                # 加权累加
                final_output[i] += weight * expert_out.squeeze()
                
        return final_output

class BasicExpert(nn.Module):
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.linear = nn.Linear(feature_in, feature_out)
    
    def forward(self, x):
        return self.linear(x)

def test_sparse_moe():
    x = torch.randn(2, 4)
    # 假设有 10 个专家，但每个样本只用 2 个
    moe = SparseMOE(4, 3, expert_number=10, top_k=2)
    out = moe(x)
    print("Output shape:", out.shape)
    print("Output:", out)

test_sparse_moe()
```

先确定expert的版本如下所示

