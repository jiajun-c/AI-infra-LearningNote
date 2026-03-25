# Kernel fusion 的triton实现

## 1. permute fusion

permute fusion指的是假设有一个tensor其shape为`[N, K, H]`，原先的操作是先将其permute为了`[N, H, K]`，进行contingous后在K维度做求和

原先的代码逻辑

`permute` + `contiguous` + `kernel`

对应的triton kernel代码如下所示

```python
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
```

假设我们希望对permute进行fusion，那么其实只需要去交换K维度和N维度即可

```python
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
```