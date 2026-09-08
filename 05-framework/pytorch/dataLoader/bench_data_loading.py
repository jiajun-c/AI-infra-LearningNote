"""
Data loading 加速实测 —— 单一脚本多组对照实验

运行：python bench_data_loading.py all
     或 python bench_data_loading.py 1 2 3 ...  单独跑某个

依赖：torch (含 CUDA)、PIL、numpy
测机：NVIDIA RTX 4060 Laptop GPU (8GB), CUDA 12.8, torch 2.9.1
"""
import io
import time
import statistics
from typing import Callable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 共用工具
# ============================================================
def _make_jpeg_bytes(H: int = 256, W: int = 256, seed: int = 0) -> bytes:
    """合成一张 jpg 的 bytes（模拟磁盘上的一张图）"""
    rng = np.random.default_rng(seed)
    arr = (rng.random((H, W, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return buf.getvalue()


def _bench(fn: Callable, n_warmup: int = 2, n_iter: int = 10) -> dict:
    """跑 n_warmup 次预热，取 n_iter 次的中位数 / 平均 / min"""
    for _ in range(n_warmup):
        fn()
    samples = []
    for _ in range(n_iter):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    return {
        "median_ms": statistics.median(samples) * 1000,
        "mean_ms":   statistics.mean(samples) * 1000,
        "min_ms":    min(samples) * 1000,
        "max_ms":    max(samples) * 1000,
    }


def _print(name: str, r: dict, extra: str = ""):
    print(f"  {name:40s} median={r['median_ms']:7.2f}ms  "
          f"mean={r['mean_ms']:7.2f}ms  min={r['min_ms']:7.2f}ms  {extra}")


# ============================================================
# 1. pin_memory vs non-pinned 的 H2D 带宽对比
# ============================================================
def bench_pin_memory_vs_nonpinned(N: int = 64, C: int = 3, H: int = 224, W: int = 224,
                                  n_iter: int = 30):
    """
    比较同一份数据，pageable vs pinned 的 H2D 拷贝带宽。
    用 cudaEvent 测单次 copy 真实耗时（排除 sync overhead）。
    """
    print("\n=== [1] pin_memory vs non-pinned H2D 带宽 ===")
    print(f"    每 tensor 大小 = {N*C*H*W*4/1024/1024:.1f} MB")
    assert torch.cuda.is_available(), "需要 CUDA"

    n_bytes = N * C * H * W * 4   # float32

    # pageable
    x_page = torch.randn(N, C, H, W)
    print(f"    x_page.is_pinned() = {x_page.is_pinned()}")

    # pinned
    x_pin = torch.empty(N, C, H, W, pin_memory=True)
    x_pin.copy_(x_page)
    print(f"    x_pin.is_pinned()  = {x_pin.is_pinned()}")

    # 用 cudaEvent 测真实 device 端时间
    def copy_cuda_event(src, non_blocking):
        """单次 copy + 用 cudaEvent 精确测 device 端耗时"""
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        # 预热
        for _ in range(3):
            _ = src.to('cuda', non_blocking=non_blocking)
        torch.cuda.synchronize()
        times = []
        for _ in range(n_iter):
            start.record()
            _ = src.to('cuda', non_blocking=non_blocking)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        return times

    t1 = copy_cuda_event(x_page, False)
    t2 = copy_cuda_event(x_page, True)
    t3 = copy_cuda_event(x_pin,   False)
    t4 = copy_cuda_event(x_pin,   True)

    def stats(times):
        return {
            "median_ms": statistics.median(times),
            "mean_ms":   statistics.mean(times),
            "min_ms":    min(times),
            "max_ms":    max(times),
        }

    r1, r2, r3, r4 = stats(t1), stats(t2), stats(t3), stats(t4)
    bw1 = n_bytes / (r1["median_ms"] / 1000) / 1e9
    bw2 = n_bytes / (r2["median_ms"] / 1000) / 1e9
    bw3 = n_bytes / (r3["median_ms"] / 1000) / 1e9
    bw4 = n_bytes / (r4["median_ms"] / 1000) / 1e9

    print(f"    (cudaEvent 测单次 H2D DMA 耗时，n_iter={n_iter})")
    _print("pageable + blocking",     r1, f"≈ {bw1:.2f} GB/s")
    _print("pageable + non_blocking", r2, f"≈ {bw2:.2f} GB/s   (⚠️  silent fallback)")
    _print("pinned   + blocking",     r3, f"≈ {bw3:.2f} GB/s")
    _print("pinned   + non_blocking", r4, f"≈ {bw4:.2f} GB/s   ← 真异步 (DMA)")

    return {"pageable_blocking": r1, "pageable_nb": r2,
            "pinned_blocking":   r3, "pinned_nb":    r4}


# ============================================================
# 2. non_blocking 真正生效 vs silent fallback 的端到端模拟
# ============================================================
def bench_nonblocking_e2e(BATCH: int = 64, N_SAMPLES: int = 50):
    """
    模拟训练循环：CPU 准备下一批 vs GPU 跑当前批，看能否 overlap。
    """
    print("\n=== [2] non_blocking 真假异步的端到端开销 ===")
    assert torch.cuda.is_available(), "需要 CUDA"

    device = 'cuda'
    H = W = 224
    pinned = torch.randn(N_SAMPLES, 3, H, W, pin_memory=True)
    page   = torch.randn(N_SAMPLES, 3, H, W)

    def fake_compute():
        # 模拟一个 kernel：让 GPU 跑一会儿（~1ms）
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        torch.cuda.synchronize()
        return (a @ b).sum().item()

    # --- 同步路径 ---
    def sync_loop():
        for i in range(N_SAMPLES):
            x = page[i:i+1].to(device, non_blocking=False)
            _ = x.sum()

    # --- 异步路径（pageable，silent fallback）---
    def async_loop_pageable():
        for i in range(N_SAMPLES):
            x = page[i:i+1].to(device, non_blocking=True)
            _ = x.sum()

    # --- 异步路径（pinned，真正异步）---
    def async_loop_pinned():
        for i in range(N_SAMPLES):
            x = pinned[i:i+1].to(device, non_blocking=True)
            _ = x.sum()

    # 预热
    sync_loop(); async_loop_pageable(); async_loop_pinned()

    r1 = _bench(sync_loop, n_warmup=1, n_iter=3)
    r2 = _bench(async_loop_pageable, n_warmup=1, n_iter=3)
    r3 = _bench(async_loop_pinned, n_warmup=1, n_iter=3)

    print(f"    (loop {N_SAMPLES} 次，per-iter 是 CPU+GPU 总时间)")
    _print("sync H2D (blocking)",   r1)
    _print("nb H2D + pageable",     r2, "← 默默退化为同步")
    _print("nb H2D + pinned",       r3, "← 真异步，可 overlap")


# ============================================================
# 3. num_workers sweep：多进程 DataLoader 加速
# ============================================================
class _SlowJpegDataset(Dataset):
    """模拟一张 jpg 解码 + resize + normalize 的延迟"""
    def __init__(self, n: int, jpeg_bytes_pool: list, transform):
        self.n = n
        self.pool = jpeg_bytes_pool
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # 模拟 disk read + jpg decode
        img = Image.open(io.BytesIO(self.pool[idx % len(self.pool)]))
        img = img.resize((224, 224))
        x = np.asarray(img, dtype=np.uint8)        # HWC, uint8
        x = torch.from_numpy(x).permute(2, 0, 1)   # CHW
        return x


def bench_num_workers_sweep(N: int = 2000, batch: int = 64):
    """
    用 0/1/2/4/8 个 worker 跑同一个慢 dataset，看吞吐。
    """
    print("\n=== [3] num_workers sweep (CPU decode bottleneck) ===")
    pool = [_make_jpeg_bytes(seed=i) for i in range(64)]
    ds = _SlowJpegDataset(N, pool, None)

    def run(num_workers, pin, persistent):
        loader = DataLoader(
            ds, batch_size=batch, num_workers=num_workers,
            pin_memory=pin, persistent_workers=persistent,
            multiprocessing_context='fork' if num_workers > 0 else None,
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        n_batches = 0
        for batch_data in loader:
            n_batches += 1
            if torch.cuda.is_available():
                batch_data = batch_data.to('cuda', non_blocking=pin)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        del loader
        return (t1 - t0), n_batches

    print(f"    dataset size={N}, batch={batch}")
    print(f"    {'workers':>8s} {'pin':>5s} {'persist':>9s} {'time(s)':>10s} {'throughput':>14s}")

    results = {}
    for nw in [0, 1, 2, 4, 8]:
        for pin in [False, True]:
            for persist in [False] if nw == 0 else [False, True]:
                # skip: workers>0 但不要 pin
                t, n = run(nw, pin, persist)
                tp = n / t
                results[(nw, pin, persist)] = (t, n, tp)
                print(f"    {nw:>8d} {str(pin):>5s} {str(persist):>9s} {t:>10.2f} {tp:>10.2f} batch/s")

    # 找最快
    best = max(results.items(), key=lambda kv: kv[1][2])
    print(f"\n    ✓ 最快配置: workers={best[0][0]}, pin={best[0][1]}, "
          f"persistent={best[0][2]} → {best[1][2]:.2f} batch/s")
    return results


# ============================================================
# 4. CPU vs GPU augmentation（torchvision v2）
# ============================================================
def bench_augment_cpu_vs_gpu(N: int = 200, batch: int = 64):
    """
    比较 CPU 端 torchvision（老 API）vs GPU 端 v2 / 手动 GPU augment。
    """
    print("\n=== [4] CPU vs GPU augmentation ===")
    assert torch.cuda.is_available(), "需要 CUDA"
    try:
        from torchvision.transforms import v2
    except ImportError:
        print("    torchvision.transforms.v2 不可用，跳过")
        return

    H = W = 224
    # 准备一张 uint8 3D tensor
    x_uint8 = torch.randint(0, 256, (batch, 3, H, W), dtype=torch.uint8)
    print(f"    input shape = {tuple(x_uint8.shape)}, dtype={x_uint8.dtype}")

    # --- CPU augment: PIL/torchvision v1 style ---
    from torchvision import transforms
    cpu_aug = transforms.Compose([
        transforms.RandomResizedCrop(H, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 把 uint8 tensor 转成 PIL list
    def cpu_augment_loop():
        for img in x_uint8:
            pil = transforms.functional.to_pil_image(img)
            _ = cpu_aug(pil)
    r1 = _bench(cpu_augment_loop, n_iter=5)

    # --- GPU augment: torchvision v2 ---
    device = 'cuda'
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1) * 255
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1) * 255
    x_gpu = x_uint8.to(device, non_blocking=True)

    gpu_aug = v2.Compose([
        v2.RandomResizedCrop(H, scale=(0.7, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(0.2, 0.2, 0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def gpu_augment_loop():
        for _ in range(N):
            _ = gpu_aug(x_gpu)
    r2 = _bench(gpu_augment_loop, n_iter=20)

    print(f"    (跑 {N} 次相同 augment，per-iter 是 {N} 次的总时间)")
    _print(f"CPU augment × {N} 次", r1)
    _print(f"GPU augment × {N} 次", r2)

    speedup = r1["median_ms"] / r2["median_ms"]
    print(f"    ✓ GPU augment 比 CPU 快约 {speedup:.1f}x")


# ============================================================
# 5. 端到端 mini 训练循环
# ============================================================
def bench_e2e_mini_train():
    """
    一个真实的小训练循环，对比：
      A) 默认 DataLoader (sync, pageable)
      B) DataLoader + pin_memory + non_blocking
      C) B + persistent_workers
    """
    print("\n=== [5] 端到端 mini 训练循环 ===")
    if not torch.cuda.is_available():
        print("    没 CUDA，跳过"); return

    N = 500
    batch = 64
    H = W = 128
    pool = [_make_jpeg_bytes(H=H, W=W, seed=i) for i in range(64)]
    ds = _SlowJpegDataset(N, pool, None)

    # 一个超级小的"模型"：一两个 op
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * H * W, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    ).to('cuda')
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    crit = torch.nn.CrossEntropyLoss()

    def train_one_epoch(num_workers, pin, persistent, nw_iter):
        loader = DataLoader(
            ds, batch_size=batch, num_workers=num_workers,
            pin_memory=pin, persistent_workers=persistent,
            multiprocessing_context='fork' if num_workers > 0 else None,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i, x in enumerate(loader):
            x = x.to('cuda', non_blocking=pin).float() / 255.0
            y = torch.randint(0, 10, (x.size(0),), device='cuda')
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if num_workers > 0:
            loader._iterator._shutdown_workers()
        return t1 - t0

    cases = [
        ("A: 0 workers, no pin",        dict(num_workers=0, pin=False, persistent=False, nw_iter=0)),
        ("B: 2 workers + pin + nb",     dict(num_workers=2, pin=True,  persistent=False, nw_iter=2)),
        ("C: 2 workers + pin + persist", dict(num_workers=2, pin=True, persistent=True,  nw_iter=2)),
        ("D: 4 workers + pin + persist", dict(num_workers=4, pin=True, persistent=True,  nw_iter=4)),
    ]
    base = None
    print(f"    {'配置':40s} {'epoch time(s)':>14s} {'speedup':>8s}")
    for name, kwargs in cases:
        t = train_one_epoch(**kwargs)
        if base is None:
            base = t
        print(f"    {name:40s} {t:>14.3f} {base/t:>7.2f}x")


# ============================================================
# 6. storage format 对比：散 jpg bytes list vs numpy memmap
# ============================================================
def bench_storage_format(N: int = 2000, batch: int = 64, H: int = 224, W: int = 224):
    """
    对比：
      A) 内存里一组 jpg bytes（散列），每次解码
      B) 预先解码好的 numpy uint8 array（memmap）
      C) 预解码 + pinned tensor
    """
    print("\n=== [6] storage format 对比 ===")

    # A) jpg bytes pool
    jpg_pool = [_make_jpeg_bytes(H=H, W=W, seed=i) for i in range(256)]

    class JpgListDS(Dataset):
        def __init__(self, n, pool):
            self.n, self.pool = n, pool
        def __len__(self): return self.n
        def __getitem__(self, i):
            return torch.from_numpy(
                np.asarray(Image.open(io.BytesIO(self.pool[i % len(self.pool)])).resize((H, W)), dtype=np.uint8)
            ).permute(2, 0, 1)

    # B) memmap 一个 npy 池
    big_arr = np.stack([
        np.asarray(Image.open(io.BytesIO(jpg_pool[i % 256])).resize((H, W)), dtype=np.uint8)
        for i in range(N)
    ])  # (N, H, W, 3)
    mm_path = '/tmp/_bench_data.npy'
    np.save(mm_path, big_arr)
    mm = np.load(mm_path, mmap_mode='r')

    class MmapDS(Dataset):
        def __init__(self, arr): self.arr = arr
        def __len__(self): return len(self.arr)
        def __getitem__(self, i):
            return torch.from_numpy(np.asarray(self.arr[i])).permute(2, 0, 1)

    # C) pinned tensor pool
    pinned_pool = torch.from_numpy(big_arr).permute(0, 3, 1, 2).contiguous().pin_memory()
    print(f"    pinned pool size = {pinned_pool.numel() * pinned_pool.element_size() / 1024 / 1024:.1f} MB")

    class PinnedDS(Dataset):
        def __init__(self, t): self.t = t
        def __len__(self): return len(self.t)
        def __getitem__(self, i): return self.t[i]

    def run(ds, num_workers, pin):
        loader = DataLoader(ds, batch_size=batch, num_workers=num_workers,
                            pin_memory=pin, persistent_workers=(num_workers > 0),
                            multiprocessing_context='fork' if num_workers > 0 else None)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        n = 0
        for x in loader:
            n += 1
            if torch.cuda.is_available() and pin:
                _ = x.to('cuda', non_blocking=True)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        if num_workers > 0:
            loader._iterator._shutdown_workers()
        return (t1 - t0, n)

    cases = [
        ("jpg bytes list, 0w, no-pin", JpgListDS(N, jpg_pool), 0, False),
        ("jpg bytes list, 2w, pin",    JpgListDS(N, jpg_pool), 2, True),
        ("numpy memmap,   0w, no-pin", MmapDS(mm),            0, False),
        ("numpy memmap,   2w, pin",    MmapDS(mm),            2, True),
        ("pinned tensor,  0w",         PinnedDS(pinned_pool), 0, True),
    ]
    print(f"    {'格式 + workers + pin':40s} {'time(s)':>10s} {'batch/s':>10s}")
    base = None
    for name, ds, nw, pin in cases:
        t, n = run(ds, nw, pin)
        if base is None: base = t
        print(f"    {name:40s} {t:>10.3f} {n/t:>10.2f} {t/base:>6.2f}x vs first")

    import os; os.remove(mm_path)


# ============================================================
# main
# ============================================================
def main():
    import sys
    benches = {
        "1": bench_pin_memory_vs_nonpinned,
        "2": bench_nonblocking_e2e,
        "3": bench_num_workers_sweep,
        "4": bench_augment_cpu_vs_gpu,
        "5": bench_e2e_mini_train,
        "6": bench_storage_format,
    }

    args = sys.argv[1:]
    if not args or args == ["all"]:
        selected = list(benches.keys())
    else:
        selected = args

    print(f"PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print("=" * 70)

    for k in selected:
        benches[k]()
        print()


if __name__ == "__main__":
    main()
