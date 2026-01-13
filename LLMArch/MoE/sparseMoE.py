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
        
        # --- 这里的实现稍微有点 trick，为了演示逻辑 ---
        # 在真实的高效 MoE 中，我们会把 x 重新排序发给专家。
        # 这里为了简单，我们还是遍历专家，但使用 mask 把它"伪装"成稀疏的，
        # 或者只计算选中的部分。
        
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