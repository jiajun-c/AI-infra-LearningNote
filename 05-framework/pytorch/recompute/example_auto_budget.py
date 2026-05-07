"""
activation_memory_budget 自动激活检查点对比实验

torch.compile 的自动激活检查点机制（源码：torch/_functorch/partitioners.py）：
  - budget=1.0  完全保留激活（默认，最快但最耗显存）
  - budget=0.x  保留约 x 比例的激活，编译器用背包算法选择重计算哪些算子
  - budget=0.0  特殊分支：只保存 node_info.inputs（图输入节点，含模型参数权重），
                中间激活全部重计算。因参数权重本身体积大，峰值不降反升，
                与手动全量 checkpoint（只丢弃激活）行为不同。

有效节省显存的区间是 (0.0, 1.0)，越小越省，但重计算开销越大。

注意：activation_memory_budget 是进程级全局变量，必须在 torch.compile 编译前设置，
且同一进程内重新编译不会重新分区。因此每个 budget 在独立子进程中测量。
预热必须跑完整前向+反向（含 requires_grad），否则编译器按无梯度图分区，
正式测量时会触发二次编译导致 budget 失效。
"""
import subprocess
import sys
import os

# ─────────────────────────────────────────────
# 子进程入口：测量单个 budget 的显存与耗时
# ─────────────────────────────────────────────
def _worker(budget: float, hidden_dim: int, batch_size: int,
            seq_len: int, num_layers: int):
    import time
    import torch
    import torch.nn as nn
    from torch._functorch import config as functorch_config

    MB = 1024 ** 2

    # 必须在 torch.compile 之前设置，AOT Autograd 分区时读取该值
    functorch_config.activation_memory_budget = budget

    class DummyBlock(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
            self.act = nn.GELU()
            self.linear2 = nn.Linear(hidden_size * 4, hidden_size)

        def forward(self, x):
            return self.linear2(self.act(self.linear1(x)))

    class LLMModel(nn.Module):
        def __init__(self, hidden_size, num_layers):
            super().__init__()
            self.layers = nn.ModuleList(
                [DummyBlock(hidden_size) for _ in range(num_layers)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    torch.cuda.empty_cache()
    model = LLMModel(hidden_dim, num_layers).cuda()
    compiled = torch.compile(model)

    x = torch.randn(batch_size, seq_len, hidden_dim,
                    requires_grad=True, device="cuda")
    target = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

    # 预热：必须跑完整前向+反向，让编译器按含梯度的图做分区，否则正式测量会二次编译
    out_w = compiled(x)
    loss_w = ((out_w - target) ** 2).mean()
    loss_w.backward()
    torch.cuda.synchronize()
    # 清掉预热产生的梯度，避免影响显存测量
    if x.grad is not None:
        x.grad = None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    out = compiled(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / MB

    # 通过 stdout 传回结果给父进程
    print(f"RESULT:{peak_mb:.2f}:{elapsed:.4f}")


# ─────────────────────────────────────────────
# 主进程：依次启动子进程并汇总结果
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 子进程模式：python example_auto_budget.py --worker <budget> <args...>
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        budget     = float(sys.argv[2])
        hidden_dim = int(sys.argv[3])
        batch_size = int(sys.argv[4])
        seq_len    = int(sys.argv[5])
        num_layers = int(sys.argv[6])
        _worker(budget, hidden_dim, batch_size, seq_len, num_layers)
        sys.exit(0)

    # ── 主进程配置 ──
    batch_size = 4
    seq_len    = 1024
    hidden_dim = 1024
    num_layers = 16

    configs = [
        (1.0,  "budget=1.0  完全保留激活"),
        (0.9,  "budget=0.9"),
        (0.8,  "budget=0.8"),
        (0.7,  "budget=0.7"),
        (0.6,  "budget=0.6"),
        (0.5,  "budget=0.5"),
        (0.4,  "budget=0.4"),
        (0.3,  "budget=0.3"),
        (0.2,  "budget=0.2"),
        (0.1,  "budget=0.1"),
        (0.0,  "budget=0.0  全部重计算"),
    ]

    print(f"batch={batch_size}, seq={seq_len}, hidden={hidden_dim}, layers={num_layers}")
    print("每个 budget 在独立子进程中运行（确保编译器重新做激活分区决策）\n")
    print(f"  {'策略':<30}  {'峰值显存':>10}  {'耗时':>8}  {'显存节省':>10}")
    print("  " + "-" * 66)

    baseline_mem = None
    results = []

    for budget, label in configs:
        cmd = [
            sys.executable, __file__, "--worker",
            str(budget), str(hidden_dim), str(batch_size),
            str(seq_len), str(num_layers),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        # 过滤出结果行，忽略 torch.compile 的 UserWarning 等
        result_line = next(
            (l for l in proc.stdout.splitlines() if l.startswith("RESULT:")), None
        )
        if result_line is None:
            print(f"  {label:<30}  子进程失败")
            if proc.stderr:
                print(proc.stderr[-500:])
            continue

        _, peak_str, elapsed_str = result_line.split(":")
        peak_mb = float(peak_str)
        elapsed = float(elapsed_str)
        results.append((label, peak_mb, elapsed))
        if baseline_mem is None:
            baseline_mem = peak_mb

    for label, peak, elapsed in results:
        saved = baseline_mem - peak
        pct   = saved / baseline_mem * 100
        print(f"  {label:<30}  {peak:>9.1f}M  {elapsed:>7.3f}s  {saved:>+8.1f}M ({pct:.1f}%)")
