"""
对比 4 种数据加载路径, 测量 DataPrefetcher 的真实收益.

对比矩阵 (固定 batch_size=32, num_workers=4, prefetch_factor=2, pin_memory=True):
  A. vanilla:           DataLoader -> .to(device, non_blocking=True)
  B. DataPrefetcher (top of base.py, 完整版, 有 record_stream)
  C. DataPrefetcher (简化版, 用户粘贴版本)
  D. DataPrefetcher 但不开 prefetch 效果 (sanity check)

每个组合跑 50 次取 median.
"""
import time
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

mp.set_sharing_strategy("file_descriptor")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
print(f"torch:  {torch.__version__}")
if torch.cuda.is_available():
    print(f"gpu:    {torch.cuda.get_device_name(0)}")
print()

# ============================================================
# Dataset / Model
# ============================================================
class SyntheticDataset(Dataset):
    """每张样本可配置 decode 延迟, 用来模拟 CPU 端开销."""
    def __init__(self, size=2048, feature_dim=224, transform_delay=0.005):
        self.size = size
        self.feature_dim = feature_dim
        self.transform_delay = transform_delay

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        data = torch.randn(3, self.feature_dim, self.feature_dim)
        label = torch.randint(0, 10, (1,)).item()
        if self.transform_delay > 0:
            time.sleep(self.transform_delay)
        return data, label


class SmallTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        enc = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# ============================================================
# 两种 DataPrefetcher 实现 (top of base.py vs 用户粘贴的简化版)
# ============================================================
class DataPrefetcherFull:
    """完整版: 顶部 base.py 那个, 没有 record_stream."""
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self.loader_iter = None
        self.next_data = None
        self.next_labels = None

    def preload(self):
        try:
            self.next_data, self.next_labels = next(self.loader_iter)
        except StopIteration:
            self.next_data = self.next_labels = None
            return
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_data = self.next_data.to(self.device, non_blocking=True)
                self.next_labels = self.next_labels.to(self.device, non_blocking=True)
        else:
            self.next_data = self.next_data.to(self.device)
            self.next_labels = self.next_labels.to(self.device)

    def __iter__(self):
        self.loader_iter = iter(self.loader)
        self.preload()
        return self

    def __next__(self):
        if self.next_data is None:
            raise StopIteration
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        data, labels = self.next_data, self.next_labels
        self.preload()
        return data, labels


class DataPrefetcherSimple:
    """简化版: 用户粘贴的版本, 带 record_stream."""
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.next_data = None
        self.next_labels = None
        self.preload()

    def preload(self):
        try:
            self.next_data, self.next_labels = next(self.loader)
        except StopIteration:
            self.next_data = self.next_labels = None
            return
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_data = self.next_data.to(self.device, non_blocking=True)
                self.next_labels = self.next_labels.to(self.device, non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        labels = self.next_labels
        if data is None:
            raise StopIteration
        if self.stream is not None:
            data.record_stream(torch.cuda.current_stream())
            labels.record_stream(torch.cuda.current_stream())
        self.preload()
        return data, labels


# ============================================================
# 训练函数 (通用)
# ============================================================
def train_step(model, optimizer, criterion, data, labels):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    return loss


def make_loader(dataset, num_workers, pin_memory, mp_ctx="fork"):
    kwargs = dict(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=pin_memory,
        timeout=60,
    )
    if num_workers > 0:
        kwargs["multiprocessing_context"] = mp.get_context(mp_ctx)
    return DataLoader(**kwargs)


def run_one(loader, model_factory, use_prefetcher, prefetch_kind, max_batches=80):
    """跑一轮训练循环, 返回 wall time (秒)."""
    model = model_factory().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()

    if use_prefetcher == "full":
        it = DataPrefetcherFull(loader, device)
    elif use_prefetcher == "simple":
        it = DataPrefetcherSimple(loader, device)
    else:
        it = loader

    n = 0
    for data, labels in it:
        if use_prefetcher not in ("full", "simple"):
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        train_step(model, optimizer, criterion, data, labels)
        n += 1
        if n >= max_batches:
            break

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.perf_counter() - t0


def bench(label, dataset, model_factory, num_workers, pin_memory,
          use_prefetcher, prefetch_kind=None, repeats=5, max_batches=80):
    times = []
    for r in range(repeats):
        loader = make_loader(dataset, num_workers, pin_memory)
        try:
            t = run_one(loader, model_factory, use_prefetcher,
                        prefetch_kind, max_batches=max_batches)
            times.append(t)
        except Exception as e:
            print(f"  [{label}] run {r} FAILED: {e}")
            return None

    times.sort()
    median = times[len(times) // 2]
    print(f"  {label:<50s} median={median:6.3f}s  "
          f"min={min(times):6.3f}s  max={max(times):6.3f}s")
    return median


# ============================================================
# 实验矩阵
# ============================================================
DATASET = SyntheticDataset(size=2048, feature_dim=224, transform_delay=0.005)
MODEL = SmallTransformerModel
REPEATS = 5
MAX_BATCHES = 80

print("=" * 78)
print("实验 1: pin_memory=True (H2D 真异步), 测 prefetcher 收益")
print("=" * 78)
print("固定: nw=4, pf=2, pin=True")
print()
baseline_t = bench(
    "A. vanilla (DataLoader + .to(non_blocking))",
    DATASET, MODEL, num_workers=4, pin_memory=True,
    use_prefetcher="none", repeats=REPEATS, max_batches=MAX_BATCHES,
)
prefetch_full_t = bench(
    "B. DataPrefetcher (full版, base.py顶部)",
    DATASET, MODEL, num_workers=4, pin_memory=True,
    use_prefetcher="full", repeats=REPEATS, max_batches=MAX_BATCHES,
)
prefetch_simple_t = bench(
    "C. DataPrefetcher (simple版, 用户粘贴)",
    DATASET, MODEL, num_workers=4, pin_memory=True,
    use_prefetcher="simple", repeats=REPEATS, max_batches=MAX_BATCHES,
)

if baseline_t and prefetch_full_t:
    print(f"\n  → Full prefetcher vs vanilla:    "
          f"{baseline_t / prefetch_full_t:.3f}x")
if baseline_t and prefetch_simple_t:
    print(f"  → Simple prefetcher vs vanilla:  "
          f"{baseline_t / prefetch_simple_t:.3f}x")
if prefetch_full_t and prefetch_simple_t:
    print(f"  → Simple vs Full:                "
          f"{prefetch_full_t / prefetch_simple_t:.3f}x")
print()

print("=" * 78)
print("实验 2: pin_memory=False (H2D 退化为同步), prefetcher 还能加速吗?")
print("=" * 78)
print("固定: nw=4, pf=2, pin=False")
print()
baseline_t2 = bench(
    "A. vanilla (pin=False)",
    DATASET, MODEL, num_workers=4, pin_memory=False,
    use_prefetcher="none", repeats=REPEATS, max_batches=MAX_BATCHES,
)
prefetch_simple_t2 = bench(
    "C. DataPrefetcher simple (pin=False)",
    DATASET, MODEL, num_workers=4, pin_memory=False,
    use_prefetcher="simple", repeats=REPEATS, max_batches=MAX_BATCHES,
)
if baseline_t2 and prefetch_simple_t2:
    print(f"\n  → Simple prefetcher vs vanilla (no pin): "
          f"{baseline_t2 / prefetch_simple_t2:.3f}x")
print()

print("=" * 78)
print("实验 3: 极轻 compute + 重 H2D, 看 prefetcher 在 H2D 瓶颈场景的表现")
print("=" * 78)
print("用一个超简单的 Linear 模型替代 transformer, 让 H2D 变成主瓶颈")
print("固定: nw=4, pf=2, pin=True, compute = 单次 Linear")
print()


class TinyModel(nn.Module):
    """极轻模型: compute 时间 < H2D 时间."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, 10)

    def forward(self, x):
        # 不要 view, 故意走大 flatten, 增加一点点 compute 但还是很少
        return self.fc(x.view(x.size(0), -1))


baseline_t3 = bench(
    "A. vanilla (compute 极轻, H2D 主导)",
    DATASET, TinyModel, num_workers=4, pin_memory=True,
    use_prefetcher="none", repeats=REPEATS, max_batches=MAX_BATCHES,
)
prefetch_simple_t3 = bench(
    "C. DataPrefetcher simple (compute 极轻)",
    DATASET, TinyModel, num_workers=4, pin_memory=True,
    use_prefetcher="simple", repeats=REPEATS, max_batches=MAX_BATCHES,
)
if baseline_t3 and prefetch_simple_t3:
    print(f"\n  → Simple prefetcher vs vanilla (compute 极轻): "
          f"{baseline_t3 / prefetch_simple_t3:.3f}x")
print()

print("=" * 78)
print("总结")
print("=" * 78)
print(f"硬件: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"batch=32, nw=4, pf=2, decode_delay=5ms (模拟CPU瓶颈)")
print()
print("DataPrefetcher 把 H2D 挪到独立 CUDA stream, 让 Copy Engine 跟 SM 并行.")
print("收益大小取决于: H2D 时间 vs compute 时间 的相对大小.")