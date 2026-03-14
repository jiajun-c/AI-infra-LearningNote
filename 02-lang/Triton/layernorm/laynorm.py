import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_tlkernel(
    X,
    Y,
    W,
    B,
    stride_X,
    stride_Y,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    rowidx = tl.program_id(0)
    row_start_ptr = X + rowidx * stride_X
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x_ptrs = row_start_ptr + offsets
    x = tl.load(x_ptrs, mask)
    xf = x.to(tl.float32)
    mean = tl.sum(xf, axis=0)/N
    x_centered = xf - mean
    var = tl.sum(x_centered*x_centered)/N
    invar = 1/tl.sqrt(var + eps)
    
    w_ptr = W + offsets
    p_ptr = B + offsets
    gamma = tl.load(w_ptr, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(p_ptr, mask=mask, other=1.0).to(tl.float32)
    y_output_ptr = Y + rowidx * stride_Y + offsets
    y = x_centered * invar * gamma + beta
    tl.store(y_output_ptr, y, mask=mask)
    
def layernorm_triton(x:torch.Tensor, weight:torch.Tensor, bias:torch.Tensor, eps: float=1e-5):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    y = torch.empty_like(x)
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    layernorm_tlkernel[(M,)](
        x, y,
        weight, bias,
        x.stride(0), y.stride(0),       
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # 用作图表 x 轴的参数名
        x_vals=[128 * i for i in range(2, 32)],  # x 轴的具体数值 (Hidden Size)
        line_arg='provider',  # 线条对应的参数名
        line_vals=['torch', 'triton'],  # 线条的标签
        line_names=['Torch', 'Triton'],  # 线条名称
        styles=[('blue', '-'), ('green', '-')],  # 样式
        ylabel='GB/s',  # y 轴标签
        plot_name='layer-norm-performance',  # 图表名称
        args={'M': 4096},  # 固定参数：Batch Size
    )
)
def benchmark(M, N, provider):
    # 准备数据
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    weight = torch.randn(N, device='cuda', dtype=torch.float32)
    bias = torch.randn(N, device='cuda', dtype=torch.float32)
    eps = 1e-5
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        # 使用 PyTorch 编译优化版本或者原生版本
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.layer_norm(x, (N,), weight, bias, eps),
            quantiles=quantiles
        )
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: layernorm_triton(x, weight, bias, eps),
            quantiles=quantiles
        )
    
    # 计算带宽 (GB/s)
    # 读取 X (4 bytes), 读取 Weight (4 bytes), 读取 Bias (4 bytes), 写入 Y (4 bytes)
    # 对于 LayerNorm，Weight 和 Bias 是广播的，相对于 X 往往可以忽略不计，
    # 但严格来说公式是：
    # Data Size = 2 * M * N * 4 (X read + Y write) + 2 * N * 4 (Weight/Bias read)
    size_bytes = 2 * M * N * x.element_size() + 2 * N * x.element_size()
    gbps = lambda ms: size_bytes / ms * 1e-6
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    # 1. 正确性验证
    torch.manual_seed(0)
    M, N = 16, 1024
    x = torch.randn(M, N, device='cuda')
    w = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    
    y_torch = torch.nn.functional.layer_norm(x, (N,), w, b)
    y_triton = layernorm_triton(x, w, b)
    
    # 比较结果
    if torch.allclose(y_torch, y_triton, atol=1e-5):
        print("✅ Correctness Check Passed!")
    else:
        print("❌ Correctness Check Failed!")
        print(f"Max Diff: {(y_torch - y_triton).abs().max()}")

    # 2. 运行 Benchmark
    print("Running benchmark...")
    benchmark.run(show_plots=False, print_data=True)