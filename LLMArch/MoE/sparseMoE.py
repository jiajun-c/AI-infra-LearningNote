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