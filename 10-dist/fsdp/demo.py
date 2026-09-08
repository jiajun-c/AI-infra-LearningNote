"""
FSDP 核心思想演示：逐层计算、用完即释放
场景：一个 4 层的大 MLP，每层权重很大，无法同时全部驻留在单卡显存中
FSDP 做法：
  - 权重按 world_size 分片存储（每卡只持有 1/N）
  - forward 到某层时，all-gather 拿到完整权重 → 计算 → 立即丢弃
  - backward 时同样 all-gather 重建权重 → 计算梯度 → reduce-scatter 只保留分片梯度
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


def reduce_scatter_param(grad_full: torch.Tensor, world_size: int) -> torch.Tensor:
    """reduce-scatter 梯度：各卡完整梯度求和后，只保留当前 rank 的分片"""
    chunk = grad_full.shape[0] // world_size
    grad_shard = torch.zeros(chunk, *grad_full.shape[1:], dtype=grad_full.dtype)
    # 把完整梯度切成 world_size 段，reduce-scatter 后 rank r 拿到「各卡第 r 段之和」
    input_list = list(torch.chunk(grad_full, world_size, dim=0))
    dist.reduce_scatter(grad_shard, input_list)
    return grad_shard


# ── 自定义 forward + backward ──────────────────────────────

class FSDPLinearFunction(torch.autograd.Function):
    """
    FSDP 风格 Linear 的自定义前向/反向：
      forward : all-gather 权重 → 计算 → 释放（只保留分片）
      backward: 重新 all-gather 权重 → 计算梯度 → reduce-scatter → 释放
    """

    @staticmethod
    def forward(ctx, x, w_shard, bias, world_size):
        # 1. all-gather：临时重建完整权重
        w_full = all_gather_param(w_shard, world_size)  # [out, in]

        # 2. 计算
        out = x @ w_full.T + bias  # [B, S, out]

        # 3. 只保存反向所需的分片权重（不缓存完整权重）
        ctx.save_for_backward(x, w_shard, bias)
        ctx.world_size = world_size
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w_shard, bias = ctx.saved_tensors
        world_size = ctx.world_size

        # 1. 重新 all-gather 权重（forward 时已释放）
        w_full = all_gather_param(w_shard, world_size)  # [out, in]

        # 2. 计算三类梯度
        grad_x = grad_out @ w_full                              # [B, S, in]
        grad_w_full = torch.einsum('...o,...i->oi', grad_out, x)  # [out, in]
        grad_bias = grad_out.reshape(-1, grad_out.shape[-1]).sum(0)  # [out]

        # 3. reduce-scatter：每卡只保留自己分片的梯度
        grad_w_shard = reduce_scatter_param(grad_w_full, world_size)  # [chunk, in]

        # 4. 释放完整权重
        del w_full

        # 返回值顺序与 forward 输入一一对应（world_size 无梯度）
        return grad_x, grad_w_shard, grad_bias, None


# ── FSDP 风格的单层 Linear ────────────────────────────────

class FSDPLinear(nn.Module):
    """
    每卡只存 w_shard = W[rank*chunk:(rank+1)*chunk, :]
    forward 时 all-gather 重建完整 W，backward 时再 all-gather 一次 + reduce-scatter
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
        # 委托给自定义 Function，走显式的 forward/backward
        return FSDPLinearFunction.apply(x, self.w_shard, self.bias, self.world_size)


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


# ── 梯度正确性验证 ────────────────────────────────────────

def verify_grad(rank, world_size, batch, seq_len, hidden):
    """
    单层 FSDPLinear，loss = out.sum()
    验证 reduce-scatter 得到的梯度 == 各卡梯度求和后的对应分片
    每张卡用不同输入，才能体现 reduce-scatter 的求和语义
    """
    torch.manual_seed(0)  # 权重确定性
    layer = FSDPLinear(hidden, hidden, rank, world_size)

    # 每张卡用不同输入（真实 FSDP 中各卡处理不同 micro-batch）
    g = torch.Generator().manual_seed(1000 + rank)
    x = torch.randn(batch, seq_len, hidden, generator=g)

    # FSDP forward + backward
    out = layer(x)
    out.sum().backward()
    grad_fsdp = layer.w_shard.grad  # [chunk, hidden]

    # 参考值：各卡本地完整梯度 → allreduce 求和 → 取本卡分片
    grad_out = torch.ones_like(out)
    grad_w_full_local = torch.einsum('...o,...i->oi', grad_out, x)
    dist.all_reduce(grad_w_full_local, op=dist.ReduceOp.SUM)
    chunk = hidden // world_size
    grad_ref = grad_w_full_local[rank * chunk: (rank + 1) * chunk]

    match = torch.allclose(grad_fsdp, grad_ref, atol=1e-4)
    print(f"[rank {rank}] 梯度校验 grad_shape={tuple(grad_fsdp.shape)}, "
          f"{'PASS' if match else 'FAIL'}")
    return match


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
    y0 = y.detach().clone()
    dist.broadcast(y0, src=0)
    match = torch.allclose(y, y0, atol=1e-5)
    print(f"[rank {rank}] forward 输出 shape={list(y.shape)}, consistency={'PASS' if match else 'FAIL'}")

    # --- backward 梯度校验 ---
    verify_grad(rank, world_size, BATCH, SEQ_LEN, HIDDEN)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
