"""
Dump torch.compile inductor 生成的 Triton kernel 代码。
运行后查看 /tmp/inductor_dump/ 目录下的输出文件。
"""

import os
import torch

# 设置 inductor 输出目录，dump 生成的代码
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCH_LOGS"] = "+output_code"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/inductor_dump"

hidden_size = 2048
seq_len = 32 * 1024
dtype = torch.bfloat16
device = "cuda"

# 创建 torch.compile(torch.nn.RMSNorm)
m = torch.nn.RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True).to(dtype=dtype, device=device)
m_compiled = torch.compile(m)

x = torch.randn(seq_len, hidden_size, dtype=dtype, device=device, requires_grad=True)

print("=" * 60)
print("Forward pass (触发编译, 生成 fwd kernel)")
print("=" * 60)
y = m_compiled(x)

print("\n" + "=" * 60)
print("Backward pass (触发编译, 生成 bwd kernel)")
print("=" * 60)
dy = torch.randn_like(y)
y.backward(dy)

print("\n" + "=" * 60)
print("Inductor 缓存目录内容:")
print("=" * 60)
for root, dirs, files in os.walk("/tmp/inductor_dump"):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            print(f"\n--- {path} ---")
