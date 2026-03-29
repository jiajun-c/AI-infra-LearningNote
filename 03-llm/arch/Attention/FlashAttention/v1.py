import torch
import math

def flashAttentionV1(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, B_r, B_c):
    seq_len, hidden_dim = Q.shape
    O = torch.zeros(seq_len, hidden_dim)
    l = torch.zeros(seq_len, 1)
    m = torch.full((seq_len, 1), -torch.inf)
    # 外层循环，遍历KV
    # Q [seq_len, hidden_dim]
    # K [seq_len, hidden_dim]
    # V [seq_len, hidden_dim]
    Tr = seq_len // B_r
    Tc = seq_len // B_c
    for j in range(Tc):
        K_j = K[j*B_c: (j+1)*B_c, :]
        V_j = V[j*B_c: (j+1)*B_c, :]
        for i in range(Tr):
            Q_i = Q[i*B_r: (i+1)*B_r, :]
            O_i = O[i*B_r: (i+1)*B_r, :]
            l_i = l[i * B_r : (i + 1) * B_r, :]
            m_i = m[i * B_r : (i + 1) * B_r, :]
            
            S_ij = Q_i @ K_j.T / (hidden_dim ** 0.5)
            m_block = torch.max(S_ij, dim=1, keepdim=True).values
            m_new = torch.maximum(m_i, m_block)
            P_ij = torch.exp(S_ij - m_new)
            exp_diff = torch.exp(m_i - m_new)
            l_new = exp_diff * l_i + torch.sum(P_ij, dim=1, keepdim=True)
            
            O_i = (O_i * exp_diff * l_i + P_ij @ V_j) / l_new
            l[i * B_r : (i + 1) * B_r, :] = l_new
            m[i * B_r : (i + 1) * B_r, :] = m_new
            O[i * B_r : (i + 1) * B_r, :] = O_i
            
    return O
                        
def standard_attention(Q, K, V):
    """标准 attention 作为参考实现"""
    seq_len, hidden_dim = Q.shape
    S = Q @ K.T / (hidden_dim ** 0.5)
    P = torch.softmax(S, dim=-1)
    O = P @ V
    return O

if __name__ == "__main__":
    seq_len = 128
    hidden_dim = 64
    B_r = 32
    B_c = 32

    torch.manual_seed(42)
    Q = torch.randn(seq_len, hidden_dim, dtype=torch.float32)
    K = torch.randn(seq_len, hidden_dim, dtype=torch.float32)
    V = torch.randn(seq_len, hidden_dim, dtype=torch.float32)

    # FlashAttention V1
    O_flash = flashAttentionV1(Q, K, V, B_r, B_c)

    # 标准 Attention
    O_standard = standard_attention(Q, K, V)

    # 比较结果
    max_diff = (O_flash - O_standard).abs().max().item()
    mean_diff = (O_flash - O_standard).abs().mean().item()
    print(f"Max absolute difference:  {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")

    if torch.allclose(O_flash, O_standard, atol=1e-5):
        print("✅ FlashAttention V1 结果与标准 Attention 一致!")
    else:
        print("❌ FlashAttention V1 结果与标准 Attention 不一致!")
