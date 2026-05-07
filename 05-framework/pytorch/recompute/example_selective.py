"""
Selective Activation Checkpointing 对比实验

三种策略：
  - 标准模式:       所有层保留激活（显存最高，计算最少）
  - 全量重计算:     所有层 checkpoint（显存最低，计算最多）
  - 选择性重计算:   每隔 k 层 checkpoint 一次（显存/计算折中）
"""
import time
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

MB = 1024 ** 2


class MemoryTracker:
    def __enter__(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.t0
        self.peak_mb = torch.cuda.max_memory_allocated() / MB


class DummyBlock(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class LLMModel(nn.Module):
    """
    ckpt_policy 控制哪些层做 checkpoint：
      "none"      — 全部保留激活
      "full"      — 全部重计算
      "selective" — 每隔 ckpt_every 层重计算一次，其余保留
    """

    def __init__(self, hidden_size=1024, num_layers=16,
                 ckpt_policy="none", ckpt_every=2):
        super().__init__()
        self.layers = nn.ModuleList([DummyBlock(hidden_size) for _ in range(num_layers)])
        self.ckpt_policy = ckpt_policy
        self.ckpt_every = ckpt_every

    def _should_ckpt(self, idx: int) -> bool:
        if self.ckpt_policy == "full":
            return True
        if self.ckpt_policy == "selective":
            # 每 ckpt_every 层中选第一层做 checkpoint，其余保留激活
            return idx % self.ckpt_every == 0
        return False

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self._should_ckpt(i):
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x


def run_once(policy, ckpt_every, hidden_dim, batch_size, seq_len, num_layers):
    torch.cuda.empty_cache()
    model = LLMModel(hidden_dim, num_layers, policy, ckpt_every).cuda()
    x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True, device="cuda")
    target = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

    # 只跑前向+反向，不含 optimizer step，隔离激活显存差异
    with MemoryTracker() as tr:
        out = model(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()

    del model, x, target, out, loss
    torch.cuda.empty_cache()
    return tr.peak_mb, tr.elapsed


if __name__ == "__main__":
    batch_size = 4
    seq_len    = 1024
    hidden_dim = 1024
    num_layers = 16

    configs = [
        ("none",      1, "标准（无重计算）"),
        ("selective", 4, "选择性 every=4（4层中1层ckpt）"),
        ("selective", 2, "选择性 every=2（2层中1层ckpt）"),
        ("full",      1, "全量重计算"),
    ]

    print(f"batch={batch_size}, seq={seq_len}, hidden={hidden_dim}, layers={num_layers}\n")
    print(f"  {'策略':<28}  {'峰值显存':>10}  {'耗时':>8}  {'显存节省':>10}")
    print("  " + "-" * 62)

    baseline_mem = None
    results = []
    for policy, every, label in configs:
        peak, elapsed = run_once(policy, every, hidden_dim, batch_size, seq_len, num_layers)
        results.append((label, peak, elapsed))
        if baseline_mem is None:
            baseline_mem = peak

    for label, peak, elapsed in results:
        saved = baseline_mem - peak
        pct   = saved / baseline_mem * 100
        print(f"  {label:<28}  {peak:>9.1f}M  {elapsed:>7.3f}s  {saved:>+8.1f}M ({pct:.1f}%)")
