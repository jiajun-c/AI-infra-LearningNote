"""
反向传播显存占用演示
观察：前向保留的激活值在反向完成后才释放
"""
import torch
import torch.nn as nn

MB = 1024 ** 2

def mem(label):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / MB
    print(f"  [{label:30s}]  已分配: {alloc:7.1f} MB")

device = "cuda"
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

# ── 模型：3 个线性层，每层输出 4096 维，batch=256 ──────────────────────────
B, D = 256, 4096
model = nn.Sequential(
    nn.Linear(D, D), nn.ReLU(),
    nn.Linear(D, D), nn.ReLU(),
    nn.Linear(D, D),
).to(device)

optimizer = torch.optim.Adam(model.parameters())

# 参数 + optimizer state 占用
mem("初始（参数）")

x = torch.randn(B, D, device=device)
mem("输入 x 分配后")

# ── 前向 ────────────────────────────────────────────────────────────────────
y = model(x)
mem("前向完成")
# 此时 GPU 上保留了：
#   - 每一层的输入激活（用于反向计算梯度）
#   - 最终输出 y

loss = y.sum()

# ── 反向 ────────────────────────────────────────────────────────────────────
print("\n  反向开始...")
loss.backward()
mem("反向完成")
# backward() 执行完毕后，各层保存的激活值已被消费并释放
# 此时 GPU 上多了：梯度张量（和参数等大）

# ── optimizer step ───────────────────────────────────────────────────────────
optimizer.step()
mem("optimizer.step 后")

optimizer.zero_grad()
mem("zero_grad 后")

print(f"\n  峰值显存: {torch.cuda.max_memory_allocated() / MB:.1f} MB")

# ── 手动演示激活值的大小 ────────────────────────────────────────────────────
print("\n── 各层激活值大小估算 ──")
# 每一层保存的激活是该层的输入，shape = (B, D)，float32 = 4 bytes
activation_size = B * D * 4 / MB
n_layers = 3
print(f"  单层激活: {B} x {D} x float32 = {activation_size:.1f} MB")
print(f"  {n_layers} 层激活合计: {activation_size * n_layers:.1f} MB")
param_size = sum(p.numel() * 4 for p in model.parameters()) / MB
print(f"  参数总量: {param_size:.1f} MB")
print(f"  梯度总量: {param_size:.1f} MB  (和参数等大)")
print(f"  Adam optimizer state: {param_size * 2:.1f} MB  (m + v，各一份)")
