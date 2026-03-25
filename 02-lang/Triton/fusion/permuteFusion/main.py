import torch
import triton
import triton.language as tl
@triton.jit
def sum_k_dimension_kernel(
    x_ptr, y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr
):
    pidm = tl.program_id(0)
    pidn = tl.program_id(1)
    
    # 1. 计算输入的指针数组 (注意使用 tl.arange)
    offset_in = pidm * N * K + pidn * K + tl.arange(0, K)
    
    # 2. 正确 Load 数据
    x = tl.load(x_ptr + offset_in)

    # 3. 在寄存器/SRAM中进行 K 维度的规约求和
    sum_val = tl.sum(x, axis=0)
    
    # 4. 修复输出偏移，精准定位到 [M, N] 的对应位置
    offset_out = pidm * N + pidn
    tl.store(y_ptr + offset_out, sum_val)

@triton.jit
def sum_k_dimension_fuse_tile_kernel(
    x_ptr,
    y_ptr,
    M, K, N,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    """
    Fused permute + tile 分块规约:
      直接从 [M, K, N] 物理布局跨步读取，分块遍历 K 维度。

    地址计算 (输入布局 [M, K, N], 行主序):
      物理地址 = m * K * N + k * N + n

    Tile 形状: (BLOCK_K, BLOCK_N) — K 在行方向, N 在列方向
    accumulator: (BLOCK_K, BLOCK_N)
    最终对 axis=0 (K方向) 求和 → (BLOCK_N,)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # accumulator 形状 (BLOCK_K, BLOCK_N), K 在行, N 在列
    accumulator = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for k_base in range(0, K, BLOCK_K):
        k_offsets = k_base + tl.arange(0, BLOCK_K)
        # (BLOCK_K, BLOCK_N): k_offsets[:, None] 行索引, n_offsets[None, :] 列索引
        offsets_in = pid_m * K * N + k_offsets[:, None] * N + n_offsets[None, :]
        mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        x_tile = tl.load(x_ptr + offsets_in, mask=mask, other=0.0)
        accumulator += x_tile
    # 对 axis=0 (K方向) 求和 → (BLOCK_N,)
    sum_val = tl.sum(accumulator, axis=0)
    offsets_out = pid_m * N + n_offsets
    mask = n_offsets < N
    tl.store(y_ptr + offsets_out, sum_val, mask=mask)

@triton.jit
def sum_k_dimension_tile_kernel(
    x_ptr,
    y_ptr,
    M, N, K,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pidm = tl.program_id(0)
    pidn = tl.program_id(1)
    n_offsets = pidn * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for k_base in range(0, K, BLOCK_K):
        k_offsets = k_base + tl.arange(0, BLOCK_K)
        offstes_in = pidm * N * K + k_offsets[None, :] + n_offsets[:, None] * K
        mask = (k_offsets[None, :] < K) & (n_offsets[:, None] < N)
        x_tile = tl.load(x_ptr + offstes_in, mask=mask)
        accumulator += x_tile
    sum_val = tl.sum(accumulator, axis=1)
    offsets_out = pidm * N  + n_offsets
    mask = offsets_out < N
    tl.store(y_ptr + offsets_out, sum_val)

@triton.jit
def sum_k_dimension_kernel_fuse_permute(
    x_ptr, y_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr
):
    pidm = tl.program_id(0)
    pidn = tl.program_id(1)
    
    # 1. 极其关键的步长计算！
    # 现在 tl.arange(0, K) 需要乘以步长 N
    k_offsets = tl.arange(0, K)
    
    # 物理地址 = M的基址 + K的跳跃地址 + N的偏移
    offset_in = (pidm * K * N) + (k_offsets * N) + pidn
    
    # 2. 跨步读取数据 (Strided Load)
    x = tl.load(x_ptr + offset_in)

    # 3. 规约求和 (x 依然是被抽取出来的 1D 向量)
    sum_val = tl.sum(x, axis=0)
    
    # 4. 写入输出 [M, N] (输出的步长是 N)
    offset_out = pidm * N + pidn
    tl.store(y_ptr + offset_out, sum_val)


# ============================================================
#  Benchmark: permute+contiguous+kernel  vs  fused kernel
# ============================================================

def bench_unfused(x_mkn, M, K, N, warmup=50, rep=200):
    """
    方案1 (Unfused):
      1) permute(0,2,1): [M,K,N] → [M,N,K]  (零拷贝，改 stride)
      2) contiguous():   真正的数据搬运 O(MNK)
      3) sum_k_dimension_kernel: 对连续的 [M,N,K] 做 K 维规约
    整体时间 = permute + contiguous + kernel
    """
    y = torch.empty(M, N, device=x_mkn.device, dtype=x_mkn.dtype)

    def fn():
        x_mnk = x_mkn.permute(0, 2, 1).contiguous()
        sum_k_dimension_kernel[(M, N)](x_mnk, y, M, N, K)
        return y

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


def bench_tile(x_mkn, M, K, N, BLOCK_N=16, BLOCK_K=16, warmup=50, rep=200):
    """
    方案2 (Tile):
      1) permute(0,2,1) + contiguous(): 数据重排
      2) sum_k_dimension_tile_kernel: 分块规约，支持任意大 K
    整体时间 = permute + contiguous + tile_kernel
    """
    y = torch.empty(M, N, device=x_mkn.device, dtype=x_mkn.dtype)
    grid = (M, triton.cdiv(N, BLOCK_N))

    def fn():
        x_mnk = x_mkn.permute(0, 2, 1).contiguous()
        sum_k_dimension_tile_kernel[grid](x_mnk, y, M, N, K, BLOCK_N, BLOCK_K)
        return y

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


def bench_fused(x_mkn, M, K, N, warmup=50, rep=200):
    """
    方案3 (Fused):
      直接在 kernel 内通过 stride 计算完成 permute，
      省掉 contiguous 的显存读写。
    整体时间 = 仅 fused_kernel
    """
    y = torch.empty(M, N, device=x_mkn.device, dtype=x_mkn.dtype)

    def fn():
        sum_k_dimension_kernel_fuse_permute[(M, N)](x_mkn, y, M, K, N)
        return y

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


def bench_fused_tile(x_mkn, M, K, N, BLOCK_N=16, BLOCK_K=16, warmup=50, rep=200):
    """
    方案4 (Fused+Tile):
      直接从 [M,K,N] 跨步读取 + 分块遍历 K 维度，
      省掉 contiguous，且支持任意大 K。
    整体时间 = 仅 fused_tile_kernel
    """
    y = torch.empty(M, N, device=x_mkn.device, dtype=x_mkn.dtype)
    grid = (M, triton.cdiv(N, BLOCK_N))

    def fn():
        sum_k_dimension_fuse_tile_kernel[grid](x_mkn, y, M, K, N, BLOCK_N, BLOCK_K)
        return y

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


def verify_correctness(M, K, N, BLOCK_N=16, BLOCK_K=16):
    """验证四种 kernel 的结果一致"""
    x_mkn = torch.randn(M, K, N, device="cuda", dtype=torch.float32)

    # PyTorch 参考
    y_ref = x_mkn.sum(dim=1)  # [M, K, N] 沿 K 求和 → [M, N]

    # 方案1: permute + contiguous + unfused kernel
    x_mnk = x_mkn.permute(0, 2, 1).contiguous()
    y1 = torch.empty(M, N, device="cuda", dtype=torch.float32)
    sum_k_dimension_kernel[(M, N)](x_mnk, y1, M, N, K)

    # 方案2: permute + contiguous + tile kernel
    y2 = torch.empty(M, N, device="cuda", dtype=torch.float32)
    grid = (M, triton.cdiv(N, BLOCK_N))
    sum_k_dimension_tile_kernel[grid](x_mnk, y2, M, N, K, BLOCK_N, BLOCK_K)

    # 方案3: fused kernel (kernel内处理stride, K作为constexpr)
    y3 = torch.empty(M, N, device="cuda", dtype=torch.float32)
    sum_k_dimension_kernel_fuse_permute[(M, N)](x_mkn, y3, M, K, N)

    # 方案4: fused + tile kernel (kernel内处理stride + 分块K)
    y4 = torch.empty(M, N, device="cuda", dtype=torch.float32)
    sum_k_dimension_fuse_tile_kernel[grid](x_mkn, y4, M, K, N, BLOCK_N, BLOCK_K)

    ok1 = torch.allclose(y1, y_ref, atol=1e-2, rtol=1e-2)
    ok2 = torch.allclose(y2, y_ref, atol=1e-2, rtol=1e-2)
    ok3 = torch.allclose(y3, y_ref, atol=1e-2, rtol=1e-2)
    ok4 = torch.allclose(y4, y_ref, atol=1e-2, rtol=1e-2)
    return ok1, ok2, ok3, ok4


if __name__ == "__main__":
    print("=" * 110)
    print(" Benchmark: Unfused vs Tile vs Fused vs FusedTile (四种 kernel 对比)")
    print("=" * 110)

    # M, K, N 全面覆盖：小/中/大规模 × 不同维度比例
    configs = [
        # ---- 小规模 (warmup / 验证) ----
        (16,   64,   16),
        (32,   128,  32),
        (64,   128,  64),
        # ---- M 固定, K 逐步放大 ----
        (128,  128,  128),
        (128,  256,  128),
        (128,  512,  128),
        (128,  1024, 128),
        (128,  2048, 128),
        (128,  4096, 128),
        # ---- K 固定, M 逐步放大 ----
        (64,   512,  128),
        (128,  512,  128),
        (256,  512,  128),
        (512,  512,  128),
        (1024, 512,  128),
        # ---- K 固定, N 逐步放大 ----
        (128,  512,  64),
        (128,  512,  128),
        (128,  512,  256),
        (128,  512,  512),
        # ---- 大规模 (接近实际 LLM 维度) ----
        (256,  1024, 256),
        (256,  2048, 256),
        (512,  1024, 256),
        (512,  2048, 2048),
    ]

    # 1. 正确性验证
    print("\n[正确性验证]")
    for M, K, N in configs:
        ok1, ok2, ok3, ok4 = verify_correctness(M, K, N)
        s1 = "✓" if ok1 else "✗"
        s2 = "✓" if ok2 else "✗"
        s3 = "✓" if ok3 else "✗"
        s4 = "✓" if ok4 else "✗"
        print(f"  (M={M:>4}, K={K:>4}, N={N:>4})  "
              f"unfused={s1}  tile={s2}  fused={s3}  fused_tile={s4}")

    # 2. 性能测试
    header = (f"{'(M, K, N)':<22} {'Unfused':>10} {'Tile':>10} {'Fused':>10} {'FusedTile':>10}"
              f" {'Tile加速':>8} {'Fused加速':>9} {'FuTile加速':>10}")
    print(f"\n{'─' * len(header)}")
    print(header)
    print(f"{'─' * len(header)}")

    for M, K, N in configs:
        x_mkn = torch.randn(M, K, N, device="cuda", dtype=torch.float32)

        ms_unfused    = bench_unfused(x_mkn, M, K, N)
        ms_tile       = bench_tile(x_mkn, M, K, N)
        ms_fused      = bench_fused(x_mkn, M, K, N)
        ms_fused_tile = bench_fused_tile(x_mkn, M, K, N)

        sp_tile       = ms_unfused / ms_tile
        sp_fused      = ms_unfused / ms_fused
        sp_fused_tile = ms_unfused / ms_fused_tile

        print(f"  ({M:>4}, {K:>4}, {N:>4})  "
              f"{ms_unfused:>8.4f}  {ms_tile:>8.4f}  {ms_fused:>8.4f}  {ms_fused_tile:>8.4f}  "
              f"{sp_tile:>6.2f}x  {sp_fused:>7.2f}x  {sp_fused_tile:>8.2f}x")

    print(f"{'─' * len(header)}")
    print("\nUnfused    = permute + contiguous + sum_k_kernel     (3步, K作为constexpr, 输入[M,N,K]连续)")
    print("Tile       = permute + contiguous + tile_kernel      (3步, 分块K, 输入[M,N,K]连续)")
    print("Fused      = fused_kernel                            (1步, kernel内跨步读[M,K,N], K作为constexpr)")
    print("FusedTile  = fused_tile_kernel                       (1步, kernel内跨步读[M,K,N], 分块K)")
    print("加速比     = Unfused / 对应方案")