"""
分别测试 in_order=False 和 persistent_workers=True 的实际效果.

策略:
  实验 A: in_order=True (默认) vs in_order=False
         - 构造"异构 worker": 让不同 worker 的 __getitem__ 耗时差异巨大
         - 这样 in_order=False 才能展示出价值 (快的 worker 先返回)
         - 测单 epoch 时间

  实验 B: persistent_workers=False (默认) vs persistent_workers=True
         - 测多 epoch 的总时间, 看 persistent 省下的 fork 成本
         - 每个 epoch 短一点 (看 worker 启动开销占比)
"""
import time
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

mp.set_sharing_strategy("file_descriptor")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}, torch={torch.__version__}")
if torch.cuda.is_available():
    print(f"gpu:    {torch.cuda.get_device_name(0)}\n")


# ============================================================
# 异构 dataset: 不同 index 用不同的延迟, 模拟 worker 速度不齐
# ============================================================
class HeterogeneousDataset(Dataset):
    """
    让 index % 4 == 0 的样本特别慢 (50ms), 其它样本快 (1ms).
    4 个 worker 平均分配 index, 所以有的 worker 摊到很多慢样本, 有的没有.
    这是测试 in_order 价值的标准场景.
    """
    def __init__(self, size=2048, slow_period=4, slow_delay=0.05, fast_delay=0.001):
        self.size = size
        self.slow_period = slow_period
        self.slow_delay = slow_delay
        self.fast_delay = fast_delay

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index % self.slow_period == 0:
            time.sleep(self.slow_delay)
        else:
            time.sleep(self.fast_delay)
        # 用一个小 tensor 即可, 不是关注点
        data = torch.randn(3, 32, 32)
        label = index % 10
        return data, label


# ============================================================
# 普通 dataset (persistent_workers 测试用)
# ============================================================
class NormalDataset(Dataset):
    def __init__(self, size=512, transform_delay=0.005):
        self.size = size
        self.transform_delay = transform_delay

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        data = torch.randn(3, 224, 224)
        label = torch.randint(0, 10, (1,)).item()
        if self.transform_delay > 0:
            time.sleep(self.transform_delay)
        return data, label


def make_loader(dataset, num_workers, in_order=None, persistent_workers=False,
                batch_size=32, pin_memory=True):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=pin_memory,
        timeout=60,
        multiprocessing_context=mp.get_context("fork"),
    )
    if in_order is not None:
        kwargs["in_order"] = in_order
    if persistent_workers:
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def bench_loader(loader, max_batches=None, warmup=2):
    """跑 loader 直到耗尽, 测 wall time. warmup 前几个 batch 跳过."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    n = 0
    for data, labels in loader:
        n += 1
        if warmup and n <= warmup:
            continue
        if max_batches and n >= max_batches:
            break
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0
    return elapsed, n


# ============================================================
# 实验 A: in_order=True vs False
# ============================================================
print("=" * 78)
print("实验 A: in_order 的影响")
print("=" * 78)
print("""
场景: 异构 dataset (1% 4 张样本慢 50ms, 其余 1ms).
      shuffle=True, 4 个 worker 平均分摊 index → 速度不均.
      in_order=True  必须等最慢的那批; in_order=False  谁快谁先返回.
""")
HETERO = HeterogeneousDataset(size=2048, slow_period=4, slow_delay=0.05, fast_delay=0.001)
EPOCHS_A = 3
REPEATS_A = 3

for in_order in [True, False]:
    times = []
    for r in range(REPEATS_A):
        loader = make_loader(HETERO, num_workers=4, in_order=in_order,
                              batch_size=32)
        # 多个 epoch, 模拟真实训练
        t0 = time.perf_counter()
        for epoch in range(EPOCHS_A):
            for data, labels in loader:
                pass  # 不算 GPU 时间, 关注数据加载
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        del loader

    times.sort()
    median = times[len(times) // 2]
    label = "in_order=True  " if in_order else "in_order=False "
    print(f"  {label}  (3 epoch total) median={median:6.3f}s  "
          f"min={min(times):6.3f}s  max={max(times):6.3f}s")

print()
print("  → in_order=False 让快 worker 先返回, 主进程不会卡在等慢 worker.")
print("  → in_order=True  严格按 send 顺序, 偶尔要等队尾.")
print()

# ============================================================
# 实验 B: persistent_workers 的影响
# ============================================================
print("=" * 78)
print("实验 B: persistent_workers 的影响")
print("=" * 78)
print("""
场景: 多 epoch 训练, 每个 epoch 短 (128 个 batch), 看 worker fork/init 的开销占比.
""")
NORMAL = NormalDataset(size=64 * 32, transform_delay=0.001)  # 64 batches
EPOCHS_B = 8
REPEATS_B = 2

for persist in [False, True]:
    times = []
    for r in range(REPEATS_B):
        loader = make_loader(NORMAL, num_workers=4, persistent_workers=persist,
                              batch_size=32)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for epoch in range(EPOCHS_B):
            for data, labels in loader:
                pass
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        del loader

    times.sort()
    median = times[len(times) // 2]
    label = "persistent=False" if not persist else "persistent=True "
    print(f"  {label}  (20 epoch total) median={median:6.3f}s  "
          f"min={min(times):6.3f}s  max={max(times):6.3f}s")

print()
print("  → persistent_workers=True  epoch 之间不 kill/fork worker, 省下 fork +")
print("    Python import + torch 初始化 + dataset 构造的开销.")
print("  → epoch 越短, persistent 收益越显著 (相对占比).")
print()

# ============================================================
# 实验 C: in_order=False + 不可重复 + skewed data 的警告展示
# ============================================================
print("=" * 78)
print("实验 C: in_order=False 的副作用 - batch 顺序不确定")
print("=" * 78)
print("""
跑两次, 看同样 80 个 batch 内 batch 内部样本组成是不是一样.
""")
HETERO2 = HeterogeneousDataset(size=512, slow_period=4, slow_delay=0.02, fast_delay=0.001)

# 收集每个 batch 的 sum (能反映 batch 组成)
def collect_batch_sums(in_order):
    sums = []
    loader = make_loader(HETERO2, num_workers=4, in_order=in_order,
                          batch_size=32)
    for i, (data, labels) in enumerate(loader):
        if i >= 40:
            break
        sums.append((int(data.sum().item()), labels.tolist()[:5]))  # 取前 5 个 label
    del loader
    return sums

sums_true1 = collect_batch_sums(True)
sums_false1 = collect_batch_sums(False)
sums_true2 = collect_batch_sums(True)
sums_false2 = collect_batch_sums(False)

print(f"  in_order=True  两次跑前 5 个 batch 的 sum: {sums_true1[:5]}")
print(f"  in_order=True  第三次跑前 5 个 batch 的 sum: {sums_true2[:5]}")
print(f"  → 两次 in_order=True 每次都基本一样 (随机种子差异除外)")
print()
print(f"  in_order=False 两次跑前 5 个 batch 的 sum: {sums_false1[:5]}")
print(f"  in_order=False 第三次跑前 5 个 batch 的 sum: {sums_false2[:5]}")
print(f"  → in_order=False 每次都不同, 且 batch 顺序随机")
print()


# ============================================================
# 总结
# ============================================================
print("=" * 78)
print("结论")
print("=" * 78)
print("""
in_order:
  - True (默认): 严格 FIFO, batch 顺序确定. 当 worker 速度差异大时, 主进程
                 会被卡在等最慢的那个, 即使其它 worker 已经准备好下一批.
  - False:  谁先完成谁先返回, 延迟更稳定, 但 batch 顺序不确定, 不利于
          reproducibility, 可能出现 skewed data distribution.
  - 用法: 当 worker 速度相对均匀 (decode 时间稳定) 时, True 和 False 差不多.
        当 worker 速度差异大 (e.g. 不同 shard 速度不同, 或 GPU aug 不同)
        时, False 能拿到更稳定的延迟.

persistent_workers:
  - False (默认): 每个 epoch 后 kill workers, 下个 epoch 重新 fork + import.
                 适合 dataset 状态会变 (e.g. dynamic dataset) 或 epoch 数少.
  - True:  workers 跨 epoch 存活, 省下 fork + import + init 开销.
          适合 dataset 静态 + epoch 数多的训练.
          ⚠️ 注意: workers 不会响应 dataset 对象的修改 (e.g. 加新样本).
""")