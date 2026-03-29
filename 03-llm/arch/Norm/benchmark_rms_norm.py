"""
Benchmark: RMS Norm - 5 种实现对比
  - Atomic: Triton kernel, tl.atomic_add 累积 dW
  - SM: Triton kernel, SM 分块本地累加
  - Inductor: 手写 Triton, 模仿 torch.compile 的两级 parallel reduce
  - torch.nn.RMSNorm: PyTorch ATen fused kernel (eager)
  - torch.compile: inductor 自动 kernel fusion
"""

import torch
import triton

from rms_norm_atomic import LigerRMSNormFunction as rms_norm_atomic_add
from rms_norm_sm import LigerRMSNormFunction as rms_norm_sm
from rms_norm_inductor import LigerRMSNormFunction as rms_norm_inductor

hidden_size = 2048


def make_torch_rmsnorm(hidden_size, dtype, device):
    m = torch.nn.RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True).to(dtype=dtype, device=device)
    return m


def make_compiled_rmsnorm(hidden_size, dtype, device):
    m = torch.nn.RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True).to(dtype=dtype, device=device)
    m = torch.compile(m)
    return m


# ===================== Benchmark =====================

configs = []
for mode in ['fwd', 'bwd']:
    configs.append(triton.testing.Benchmark(
        x_names=['seq_lens'],
        x_vals=[32*1024, 64*1024, 128*1024],
        line_arg='provider',
        line_vals=['atomic', 'sm', 'inductor', 'torch', 'compile'],
        line_names=['Atomic', 'SM', 'Inductor-style', 'torch.nn.RMSNorm', 'torch.compile'],
        styles=[('green', '-'), ('blue', '-'), ('purple', '-'), ('red', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name=f'{mode}-hidden_size_{hidden_size}-bf16',
        args={'hidden_size': hidden_size, 'dtype': torch.bfloat16, 'mode': mode}
    ))


@triton.testing.perf_report(configs)
def benchmark(seq_lens, hidden_size, dtype, mode, provider, device='cuda'):
    torch.manual_seed(42)
    x = torch.randn(seq_lens, hidden_size, dtype=dtype, device=device, requires_grad=True)

    def get_mem_io(tensor):
        return tensor.numel() * tensor.element_size()
    total_bytes = sum(get_mem_io(tensor) for tensor in [x]) * 2

    if 'atomic' == provider:
        weight = torch.randn(hidden_size, dtype=dtype, device=device, requires_grad=True)
        fn = lambda: rms_norm_atomic_add.apply(x, weight)
    elif 'sm' == provider:
        weight = torch.randn(hidden_size, dtype=dtype, device=device, requires_grad=True)
        fn = lambda: rms_norm_sm.apply(x, weight)
    elif 'inductor' == provider:
        weight = torch.randn(hidden_size, dtype=dtype, device=device, requires_grad=True)
        fn = lambda: rms_norm_inductor.apply(x, weight)
    elif 'compile' == provider:
        m = make_compiled_rmsnorm(hidden_size, dtype, device)
        fn = lambda: m(x)
        for _ in range(3):
            y_warmup = fn()
            if mode == 'bwd':
                dy_warmup = torch.randn_like(y_warmup)
                torch.autograd.backward(tensors=[y_warmup], grad_tensors=[dy_warmup])
        torch.cuda.synchronize()
    else:
        m = make_torch_rmsnorm(hidden_size, dtype, device)
        fn = lambda: m(x)

    if 'fwd' == mode:
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    else:
        y = fn()
        dy = torch.randn_like(y)
        fn_bwd = lambda: torch.autograd.backward(tensors=[y], grad_tensors=[dy], retain_graph=True)
        ms, min_ms, max_ms = triton.testing.do_bench(fn_bwd, quantiles=[0.5, 0.2, 0.8])

    gbps = lambda ms: total_bytes / ms * 1e-6

    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    benchmark.run(save_path='results', print_data=True)
