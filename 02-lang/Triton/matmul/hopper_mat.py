import triton
import triton.language as tl


def _hopper_configs():
    # (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
    # Hopper 上 TMA + WGMMA 适合更大的 tile 和更深的流水线
    configs = [
        (128, 256, 64, 3, 8),
        (256, 128, 64, 3, 8),
        (256,  64, 64, 4, 4),
        (128, 128, 64, 4, 4),
        ( 64, 128, 64, 4, 4),
        (128,  64, 64, 4, 4),
        ( 64,  64, 64, 5, 2),
    ]
    return [
        triton.Config(
            {'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk},
            num_stages=ns, num_warps=nw,
        )
        for bm, bn, bk, ns, nw in configs
    ]


@triton.autotune(configs=_hopper_configs(), key=['M', 'N', 'K'])
@triton.jit
def hopper_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # ----------------------------------------------------------------
    # 1. 构造 TMA 块指针 (Block Pointers)
    #    与 normal_mat 的 arange + mask 方案对比：
    #    - 无需手动计算 offset、无需 mask，boundary_check 自动处理越界
    #    - 底层由 Hopper TMA 硬件引擎执行 cp.async.bulk.tensor 搬运
    # ----------------------------------------------------------------
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),  # 内层维度 K 在物理上连续 → 对齐 TMA 搬运
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),  # 内层维度 N 在物理上连续
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ----------------------------------------------------------------
    # 2. K 维度循环: TMA 异步加载 + WGMMA 计算
    #    - tl.load(block_ptr) → cp.async.bulk.tensor (TMA)
    #    - tl.dot(a, b, acc)  → wgmma 指令 (Warp Group MMA)
    #    - tl.advance()       → 指针偏移，无额外地址运算
    #    - num_stages 控制流水线深度，编译器自动插入 barrier
    # ----------------------------------------------------------------
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # ----------------------------------------------------------------
    # 3. 结果写回 (同样用 block_ptr，底层走 TMA store)
    # ----------------------------------------------------------------
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))
