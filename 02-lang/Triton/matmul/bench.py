import torch
import triton
from normal_mat import simple_matmul
from hopper_mat import hopper_matmul_kernel


def triton_matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    simple_matmul[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def hopper_matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    hopper_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# ---------- 正确性验证 ----------
def test_correctness():
    M, N, K = 512, 512, 512
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    torch_out = torch.matmul(a, b)

    # normal matmul
    triton_out = triton_matmul(a, b)
    if torch.allclose(triton_out, torch_out, atol=1e-1, rtol=1e-2):
        print("✅ normal_mat  correctness passed!")
    else:
        max_diff = (triton_out - torch_out).abs().max().item()
        print(f"❌ normal_mat  correctness failed! max diff = {max_diff}")

    # hopper matmul
    hopper_out = hopper_matmul(a, b)
    if torch.allclose(hopper_out, torch_out, atol=1e-1, rtol=1e-2):
        print("✅ hopper_mat  correctness passed!")
    else:
        max_diff = (hopper_out - torch_out).abs().max().item()
        print(f"❌ hopper_mat  correctness failed! max diff = {max_diff}")


# ---------- 性能测试 ----------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=[128 * i for i in range(2, 33)],  # 256 ~ 4096
        line_arg='provider',
        line_vals=['triton', 'hopper', 'cublas'],
        line_names=['Triton (normal)', 'Triton (hopper)', 'cuBLAS (torch)'],
        styles=[('blue', '-'), ('red', '-'), ('green', '-')],
        ylabel='TFLOPS',
        plot_name='matmul-performance',
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_matmul(a, b), quantiles=quantiles
        )
    elif provider == 'hopper':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: hopper_matmul(a, b), quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), quantiles=quantiles
        )

    # TFLOPS = 2*M*N*K / (ms * 1e-3) / 1e12
    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


if __name__ == '__main__':
    test_correctness()
    benchmark.run(show_plots=False, print_data=True)
