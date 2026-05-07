"""
使用自定义 Module（带显式 forward/backward）对比
gradient checkpoint 开启与关闭时的显存占用差异
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

MB = 1024 ** 2

def mem(label):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / MB
    print(f"  [{label:35s}]  已分配: {alloc:7.1f} MB")


# ── 自定义模块：带 forward，autograd 自动处理 backward ─────────────────────────
class Block(nn.Module):
    """单个 Transformer-like 块：Linear → LayerNorm → GELU → Linear"""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保存中间激活供 autograd 反向使用
        h = self.fc1(x)       # (B, dim*4)
        h = self.act(h)
        h = self.fc2(h)       # (B, dim)
        return self.norm(x + h)


class DeepModel(nn.Module):
    def __init__(self, dim: int, n_blocks: int, use_checkpoint: bool):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([Block(dim) for _ in range(n_blocks)])
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                # checkpoint 不保存中间激活，反向时重新计算 forward
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.head(x).squeeze(-1)


def run(use_checkpoint: bool):
    tag = "checkpoint" if use_checkpoint else "普通"
    print(f"\n{'='*55}")
    print(f"  模式: {tag}")
    print(f"{'='*55}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    B, D, N = 256, 1024, 24  # batch=256, dim=1024, 24 个 Block（激活占比更大）

    model = DeepModel(dim=D, n_blocks=N, use_checkpoint=use_checkpoint).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    x = torch.randn(B, D, device="cuda")

    mem("初始（参数已上 GPU）")

    # ── 前向 ──────────────────────────────────────────────────────────────────
    y = model(x)
    loss = y.mean()
    mem("前向完成")
    peak_fwd = torch.cuda.memory_allocated() / MB  # 激活峰值在此处

    # ── 反向 ──────────────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()   # 隔离反向阶段峰值
    loss.backward()
    peak_bwd = torch.cuda.max_memory_allocated() / MB
    mem("反向完成")

    optimizer.step()
    optimizer.zero_grad()
    mem("optimizer step + zero_grad 后")

    print(f"\n  前向激活峰值: {peak_fwd:.1f} MB")
    print(f"  反向阶段峰值: {peak_bwd:.1f} MB")
    return peak_fwd, peak_bwd


if __name__ == "__main__":
    fwd_normal, bwd_normal = run(use_checkpoint=False)
    fwd_ckpt,   bwd_ckpt   = run(use_checkpoint=True)

    print(f"\n{'='*60}")
    print(f"  {'':20s}  {'普通':>10s}  {'checkpoint':>10s}  {'差值':>10s}")
    print(f"  {'前向激活峰值':20s}  {fwd_normal:>9.1f}M  {fwd_ckpt:>9.1f}M"
          f"  {fwd_normal-fwd_ckpt:>+9.1f}M")
    print(f"  {'反向阶段峰值':20s}  {bwd_normal:>9.1f}M  {bwd_ckpt:>9.1f}M"
          f"  {bwd_normal-bwd_ckpt:>+9.1f}M")
    print(f"\n  checkpoint 前向省了 {fwd_normal-fwd_ckpt:.1f} MB，"
          f"反向多用了 {bwd_ckpt-bwd_normal:.1f} MB（重算激活的代价）")
    print(f"{'='*60}")
