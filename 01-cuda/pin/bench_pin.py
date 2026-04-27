"""
PyTorch pinned memory 分配策略耗时对比
测试以下三种方式：
  1. torch.empty(..., pin_memory=True)         — PyTorch 直接分配 pinned memory
  2. torch.empty(...) + tensor.pin_memory()    — 先普通分配再 pin
  3. numpy (malloc + madvise hugepage) + from_numpy + pin_memory()  — 大页预fault再 pin

运行: python bench_pin.py
依赖: torch, numpy
"""

import time
import ctypes
import numpy as np
import torch

libc = ctypes.CDLL("libc.so.6", use_errno=True)

MADV_HUGEPAGE = 14  # linux/mman.h

def madvise_hugepage(ptr: int, size: int):
    ret = libc.madvise(ctypes.c_void_p(ptr), ctypes.c_size_t(size), ctypes.c_int(MADV_HUGEPAGE))
    if ret != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"madvise failed: {errno}")

def now_ms() -> float:
    return time.perf_counter() * 1e3

HUGE = 2 * 1024 * 1024  # 2 MB

# ── 方式 1：torch.empty pin_memory=True ─────────────────────────────────────
def bench_torch_direct(size_bytes: int, label: str):
    torch.cuda.synchronize()
    t0 = now_ms()
    t = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
    alloc_ms = now_ms() - t0
    print(f"  direct      {label:<10s}  alloc+pin: {alloc_ms:8.2f} ms")
    del t

# ── 方式 2：torch.empty + .pin_memory() ─────────────────────────────────────
def bench_torch_pin(size_bytes: int, label: str):
    torch.cuda.synchronize()
    t0 = now_ms()
    t = torch.empty(size_bytes, dtype=torch.uint8)
    alloc_ms = now_ms() - t0

    t0 = now_ms()
    tp = t.pin_memory()
    pin_ms = now_ms() - t0

    print(f"  malloc+pin  {label:<10s}  malloc:    {alloc_ms:8.2f} ms  pin: {pin_ms:8.2f} ms  total: {alloc_ms+pin_ms:8.2f} ms")
    del t, tp

# ── 方式 3：numpy hugepage prefault + from_numpy + .pin_memory() ─────────────
#
# numpy 底层用 malloc，通过 madvise(MADV_HUGEPAGE) 让内核分配 2MB 大页，
# 再手动写每个大页触发 fault，最后 from_numpy 零拷贝包装，pin_memory() 锁页。
# 对比方式 2 可以看出大页对 pin 阶段的加速效果。
def bench_numpy_hugepage_pin(size_bytes: int, label: str, nthreads: int = 4):
    import concurrent.futures

    # posix_memalign 保证 2MB 对齐，madvise 要求地址对齐到页边界
    n_elem = (size_bytes + HUGE - 1) // HUGE * HUGE
    raw = ctypes.create_string_buffer(n_elem + HUGE)  # 多留一个 HUGE 用于手动对齐
    addr = ctypes.addressof(raw)
    aligned_addr = (addr + HUGE - 1) & ~(HUGE - 1)
    buf = np.frombuffer(
        (ctypes.c_char * n_elem).from_address(aligned_addr), dtype=np.uint8
    )

    t0 = now_ms()
    # 通知内核用 THP，地址已对齐
    madvise_hugepage(aligned_addr, n_elem)

    # 多线程写每个 2MB 大页一次（触发实际内存分配）
    chunk = n_elem // nthreads
    def prefault(tid):
        start = tid * chunk
        end = n_elem if tid == nthreads - 1 else start + chunk
        for p in range(start, end, HUGE):
            buf[p] = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads) as ex:
        list(ex.map(prefault, range(nthreads)))
    fault_ms = now_ms() - t0

    t0 = now_ms()
    t = torch.from_numpy(buf[:size_bytes])  # 零拷贝
    tp = t.pin_memory()
    pin_ms = now_ms() - t0

    print(f"  hugepage    {label:<10s}  fault:     {fault_ms:8.2f} ms  pin: {pin_ms:8.2f} ms  total: {fault_ms+pin_ms:8.2f} ms  (threads={nthreads})")
    del raw, buf, t, tp

# ── 方式 4：cudaMallocHost（通过 torch allocator 间接，baseline 对比） ────────
#
# torch.cuda.caching_allocator_alloc 直接调用 cudaMallocHost，
# 内部一次调用完成分配+pin，是最底层的 baseline。
def bench_cuda_malloc_host(size_bytes: int, label: str):
    torch.cuda.synchronize()
    t0 = now_ms()
    ptr = torch.cuda.caching_allocator_alloc(size_bytes, device=0)
    alloc_ms = now_ms() - t0
    print(f"  cudaMallocH {label:<10s}  alloc:     {alloc_ms:8.2f} ms")
    torch.cuda.caching_allocator_delete(ptr)

def main():
    # 预热 CUDA context
    _ = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

    MB = 1024 * 1024
    cases = [
        (64   * MB,  "64 MB"),
        (256  * MB,  "256 MB"),
        (1024 * MB,  "1024 MB"),
        (4096 * MB,  "4096 MB"),
    ]

    header = f"  {'method':<12s}  {'size':<10s}  {'timing'}"
    sep    = "  " + "-" * 90

    for size, label in cases:
        print(f"\n[{label}]")
        print(header)
        print(sep)
        bench_torch_direct(size, label)
        bench_torch_pin(size, label)
        bench_numpy_hugepage_pin(size, label, nthreads=4)
        bench_cuda_malloc_host(size, label)

if __name__ == "__main__":
    main()
