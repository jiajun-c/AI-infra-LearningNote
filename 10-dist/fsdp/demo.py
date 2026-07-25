"""
FSDP 核心思想演示：逐层计算、用完即释放
场景：一个 4 层的大 MLP，每层权重很大，无法同时全部驻留在单卡显存中
FSDP 做法：
  - 权重按 world_size 分片存储（每卡只持有 1/N）
  - forward 到某层时，all-gather 拿到完整权重 → 计算 → 立即丢弃
  - backward 时同样 all-gather 重建权重 → 计算梯度 → 再次丢弃，只保留分片梯度
"""

import torch
import torch.distributed as dist
import torch.nn as nn


# ── 工具函数 ──────────────────────────────────────────────

def shard_param(param: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    """将完整权重按第0维均匀切分，返回当前 rank 的分片"""
    assert param.shape[0] % world_size == 0
    chunk = param.shape[0] // world_size
    return param[rank * chunk: (rank + 1) * chunk].detach().clone()


def all_gather_param(shard: torch.Tensor, world_size: int) -> torch.Tensor:
    """all-gather 还原完整权重（仅用于当前层计算，之后丢弃）"""
    gathered = [torch.zeros_like(shard) for _ in range(world_size)]
    dist.all_gather(gathered, shard)
    return torch.cat(gathered, dim=0)


# ── FSDP 风格的单层 Linear ────────────────────────────────

class FSDPLinear(nn.Module):
    """
    每卡只存 w_shard = W[rank*chunk:(rank+1)*chunk, :]
    forward 时 all-gather 重建完整 W，计算后立即释放
    """
    def __init__(self, in_features: int, out_features: int, rank: int, world_size: int):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.in_features = in_features
        self.out_features = out_features

        # 初始化完整权重，然后只保留本卡分片
        full_w = torch.randn(out_features, in_features) * 0.01
        self.w_shard = nn.Parameter(shard_param(full_w, world_size, rank))
        # bias 不分片，每卡保存完整（也可以分片，此处简化）
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. all-gather：临时重建完整权重
        w_full = all_gather_param(self.w_shard, self.world_size)  # [out, in]

        # 2. 计算
        out = x @ w_full.T + self.bias  # [B, S, out]

        # 3. w_full 是局部变量，函数返回后自动释放（不存入 self）
        return out


# ── 演示模型 ──────────────────────────────────────────────

class BigMLP(nn.Module):
    """4 层大 MLP，用 FSDPLinear 逐层计算"""
    def __init__(self, hidden: int, rank: int, world_size: int):
        super().__init__()
        self.layers = nn.ModuleList([
            FSDPLinear(hidden, hidden, rank, world_size)
            for _ in range(4)
        ])
        self.act = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.act(layer(x))
        return x


# ── 主流程 ────────────────────────────────────────────────

def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    BATCH = 4
    SEQ_LEN = 16
    HIDDEN = 1024  # 每层权重 [1024, 1024]，每卡只存 [1024/N, 1024]

    torch.manual_seed(42)

    # --- 构建 FSDP 模型 ---
    model = BigMLP(HIDDEN, rank, world_size)

    # --- 显存对比（单位 MB）---
    # DP:   每卡加载全部 4 层完整权重 = 4 * 1024*1024 * 4B = 16 MB
    # FSDP: 每卡只存分片             = 16 MB / world_size
    shard_mb = sum(p.numel() * 4 for p in model.parameters()) / 1e6
    if rank == 0:
        full_mb = shard_mb * world_size
        print(f"[显存对比] DP 每卡: {full_mb:.1f} MB | FSDP 每卡: {shard_mb:.1f} MB (节省 {world_size}x)")

    # --- forward ---
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    y = model(x)

    # --- 正确性验证：所有卡输出应一致（权重通过 all-gather 还原，结果确定性一致）---
    # 收集 rank0 的输出广播给所有卡做对比
    y0 = y.detach().clone()
    dist.broadcast(y0, src=0)
    match = torch.allclose(y, y0, atol=1e-5)

    print(f"[rank {rank}] output shape={list(y.shape)}, consistency={'PASS' if match else 'FAIL'}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
