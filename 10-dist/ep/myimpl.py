import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(d_model, d_ff))
        self.w2 = nn.Parameter(torch.randn(d_ff, d_model))

    def forward(self, x: torch.Tensor):
        return F.relu(x @  self.w1) @ self.w2

class Gate(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_model,num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x @ self.w, dim=-1)

class EP(nn.Module):
    def __init__(self, num_experts:int, topk:int, d_model: int, d_ff:int, world_size: int):
        self.num_experts = num_experts
        self.topk = topk
        self.gate = Gate(d_model, num_experts)
        self.num_expert_pre_rank = num_experts // world_size
        self.world_size = world_size
        self.experts = [Expert(d_model, d_ff) for _ in range(self.num_experts)]
    def forward(self, x: torch.Tensor):

        # gate
        top_idx, top_w = [], []
        for r in range(self.world_size):
            probs_r = self.gate(x)
            w_r, idx_r = torch.topk(probs_r, self.topk, dim=-1)
            w_r = w_r / w_r.sum(-1, keepdim=True)
            top_w.append(w_r)
            top_idx.append(idx_r)

        _, T, D = x.shape
        recv = [[[] for _ in range(self.num_expert_pre_rank)] for _ in range(self.world_size)]

        # dispatch
        for src in range(self.world_size):
            for k in range(self.topk):
                for tok_idx in range(T):
                    e = idx_r[src][top_idx, k]
                    dst = e // self.num_expert_pre_rank
                    local_e =e % self.num_expert_pre_rank
                    recv[dst][local_e].append((src, tok_idx, float(top_w[src][tok_idx, k]), x[src][tok_idx]))


        # compute
        results = [[[] for _ in range(self.num_expert_pre_rank)] for _ in range(self.world_size)]:
        for src in range(self.world_size):
            for local_e in range(self.num_expert_pre_rank):
                group = recv[src][local_e]
                if not group:
                    continue
                e = src * self.num_expert_pre_rank + local_e
                rows = torch.stack([g[3] for g in group])
                y = self.experts[e](rows)
                p = torch.Tensor([[g[2] for g in group]], dtype=rows.dtype)
                for j, (src, tok_i, _, _) in enumerate(group):
                    results[src][local_e].append((src, tok_i, y[j] * p[j]))

        # combine
        out = torch.zeros_like(x)
        for rank in range(self.world_size):
            for local_e in range(self.num_expert_pre_rank):
                for src, tok_i, y_row in results[rank][local_e]:
                    out[src][tok_i] += y_row

        return out


testExpert = Expert(1024, 128)
x = torch.randn(248, 1024)
print(testExpert(x))