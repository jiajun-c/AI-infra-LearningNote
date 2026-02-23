import torch
import torch.nn.functional as F
import math
import time

def format_mem(memory_in_bytes):
"""将字节转换为 MB"""
return f"{memory_in_bytes / (1024 ** 2):.2f} MB"

def profile_standard_prefill(x, Wq, Wk, Wv):
print("--- 运行 标准无分块 Prefill ---")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

start_time = time.time()

_, seq_len, d_model = x.shape
# x (seq, d_model)
# Wq (d_model, d_model)
q = x @ Wq
k = x @ Wk
v = x @ Wv

# q (1, seq, d_model)
# k (1, seq, d_model)
# v (1, seq, d_model)
# (1, seq, d_model) * (1, d_model, seq) = (1, seq, seq)
# 这里的 scores 矩阵会产生巨大的显存峰值 O(N^2)
scores = q @ k.transpose(-2, -1) / math.sqrt(d_model)

mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)
mask = torch.tril(mask)
scores.masked_fill_(~mask, float('-inf'))

attn_weights = F.softmax(scores, dim=-1)
out = attn_weights @ v

torch.cuda.synchronize()
end_time = time.time()

peak_mem = torch.cuda.max_memory_allocated()
print(f"耗时: {(end_time - start_time) * 1000:.2f} ms")
print(f"显存峰值: {format_mem(peak_mem)}\n")
del q, k, v, scores, mask, attn_weights, out
def profile_chunked_prefill(x, Wq, Wk, Wv, chunk_size):
print(f"--- 运行 Chunked Prefill (Chunk Size: {chunk_size}) ---")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

start_time = time.time()

_, seq_len, d_model = x.shape
global_k_cache = []
global_v_cache = []

for start_idx in range(0, seq_len, chunk_size):
    end_idx = min(start_idx + chunk_size, seq_len)
    current_chunk_len = end_idx - start_idx
    
    chunk_x = x[:, start_idx:end_idx, :] 
    
    q_chunk = chunk_x @ Wq
    k_chunk = chunk_x @ Wk
    v_chunk = chunk_x @ Wv
    
    global_k_cache.append(k_chunk)
    global_v_cache.append(v_chunk)
    
    past_and_present_k = torch.cat(global_k_cache, dim=1)
    past_and_present_v = torch.cat(global_v_cache, dim=1)
    total_len_so_far = past_and_present_k.size(1)

    # 这里的 scores 矩阵每次最大只有 Chunk_Size * Seq_Len
    scores = q_chunk @ past_and_present_k.transpose(-2, -1) / math.sqrt(d_model)
    
    mask = torch.ones((current_chunk_len, total_len_so_far), dtype=torch.bool, device=x.device)
    mask = torch.tril(mask, diagonal=total_len_so_far - current_chunk_len)
    scores.masked_fill_(~mask, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    out = attn_weights @ past_and_present_v
    
torch.cuda.synchronize()
end_time = time.time()

peak_mem = torch.cuda.max_memory_allocated()
print(f"耗时: {(end_time - start_time) * 1000:.2f} ms")
print(f"显存峰值: {format_mem(peak_mem)}\n")
del global_k_cache, global_v_cache, past_and_present_k, past_and_present_v, scores, mask, attn_weights, out
if name == "main":
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
print("警告: 未检测到 GPU，无法分析显存峰值。")
else:
# 参数设置：增加 seq_len 以凸显 O(N^2) 的显存压力
seq_len = 8192
chunk_size = 512
d_model = 2048

    print(f"测试配置: Seq_Len={seq_len}, D_Model={d_model}, Chunk_Size={chunk_size}\n")
    
    # 初始化张量并移至 GPU
    x = torch.randn(1, seq_len, d_model, device=device)
    Wq = torch.randn(d_model, d_model, device=device)
    Wk = torch.randn(d_model, d_model, device=device)
    Wv = torch.randn(d_model, d_model, device=device)

    profile_chunked_prefill(x, Wq, Wk, Wv, chunk_size)
    profile_standard_prefill(x, Wq, Wk, Wv)