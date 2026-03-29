import triton
import triton.language as tl
import torch

def _matmul_configs():
    # (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
    configs = [
        (128, 256, 64, 3, 8),
        ( 64, 256, 32, 4, 4),
        (128, 128, 32, 4, 4),
        (128,  64, 32, 4, 4),
        ( 64, 128, 32, 4, 4),
        (128,  32, 32, 4, 4),
        ( 64,  32, 32, 5, 2),
        ( 32,  64, 32, 5, 2),
    ]
    return [
        triton.Config(
            {'BLOCK_SIZE_M': bm, 'BLOCK_SIZE_N': bn, 'BLOCK_SIZE_K': bk},
            num_stages=ns, num_warps=nw,
        )
        for bm, bn, bk, ns, nw in configs
    ]


@triton.autotune(configs=_matmul_configs(), key=['M', 'N', 'K'])
@triton.jit
def simple_matmul(A_ptr: torch.Tensor,
                  B_ptr: torch.Tensor,
                  C_Ptr: torch.Tensor,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr,
                  ):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        offset_k = k + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = A_ptr + offset_m[:, None]*stride_am + offset_k * stride_ak
        b_ptrs = B_ptr + offset_k[:, None]*stride_bk + offset_n * stride_bn
        a = tl.load(a_ptrs, mask=offset_k[None, :] < K, other=0.0)
        b = tl.load(b_ptrs, mask=offset_k[:, None] < K, other=0.0)
        acc += tl.dot(a, b)
    c = acc.to(tl.float16)
    c_ptrs = C_Ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
    c_mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)