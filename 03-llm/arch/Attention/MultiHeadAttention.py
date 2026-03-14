import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head) -> None:
        super().__init__()
        self.nums_head = nums_head
        self.head_dim = hidden_dim//nums_head
        self.hidden_dim = hidden_dim

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.att_dropout = nn.Dropout(0.1)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, attention_mask=None):
        batch_size, seq_len, _ = X.size()

        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.k_proj(X)

        q_state = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        k_state = K.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        v_state = V.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)

        attention_weight = (
            q_state@k_state.transpose(-1, -2)/math.sqrt(self.head_dim)
        )
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0, float("-1e20")
            )
        attention_weight = torch.softmax(attention_weight, dim=3)

        # batch_size, seq_len, nums_head, nums_head
        attention_weight = self.att_dropout(attention_weight)
        output_mid = attention_weight @ v_state
        # batch_size, seq_len, nums_head, head_dim
        output_mid = output_mid.transpose(1, 2).contiguous()
        output = output_mid.view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        return output


attention_mask = (
    torch.tensor(
        [
            [0, 1],
            [0, 0],
            [1, 0],
        ]
    )
    .unsqueeze(1)
    .unsqueeze(2)
    .expand(3, 8, 2, 2)
)

x = torch.rand(3, 2, 128)
net = MultiHeadAttention(128, 8)
net(x, attention_mask).shape


