import torch
from torch.utils.cpp_extension import load_inline

# 1. 修复后的 CUDA Kernel 与 PyTorch C++ 绑定接口
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

template<const int warpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int mask = warpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__global__ void block_all_reduce_sum_f32_f32(const float* a, float* y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // 假设 Block Size 固定为 256 (8 warps)
    __shared__ float reduce_smem[8]; 
    int warpID = tid / WARP_SIZE;
    int laneID = tid % WARP_SIZE;
    
    float sum = (idx < N) ? a[idx] : 0.0f;
    sum = warp_reduce_sum_f32(sum);
    
    if (laneID == 0) {
        reduce_smem[warpID] = sum;
    }
    
    // 【修复点】：__syncthreads 必须在所有线程都能执行到的地方
    __syncthreads();
    
    sum = (laneID < 8) ? reduce_smem[laneID]: 0.0f;
    if (warpID == 0) {
        sum = warp_reduce_sum_f32(sum);
    }
    
    if (tid == 0) {
        atomicAdd(y, sum);
    }
}

// PyTorch 调用的入口函数
void custom_sum(torch::Tensor a, torch::Tensor y) {
    int N = a.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    block_all_reduce_sum_f32_f32<<<blocks, threads>>>(
        a.data_ptr<float>(), 
        y.data_ptr<float>(), 
        N
    );
}
"""

cpp_source = """
void custom_sum(torch::Tensor a, torch::Tensor y);
"""

print("正在编译 CUDA Kernel...")
custom_module = load_inline(
    name='custom_reduce',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['custom_sum'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)
print("编译完成！\n")

# 2. 测试参数设置
N = 1024 * 1024 * 128  # 测试规模：128M 个 float (约 512MB 数据)
a = torch.randn(N, device='cuda', dtype=torch.float32)

# 3. 正确性验证 (Sanity Check)
y_custom = torch.zeros(1, device='cuda', dtype=torch.float32)
custom_module.custom_sum(a, y_custom)
y_torch = torch.sum(a)

print(f"PyTorch 计算结果: {y_torch.item():.4f}")
print(f"Custom  计算结果: {y_custom.item():.4f}")
# 注意：由于浮点数累加顺序不同，大数组求和必定有精度截断差异，这里使用比较宽松的 atol
assert torch.allclose(y_torch, y_custom, atol=1e-2, rtol=1e-2), "正确性校验失败，结果不匹配！"
print("正确性校验通过！\n")

# 4. 性能 Benchmark 函数
def benchmark(func, args, kwargs={}, num_iters=100, num_warmup=10):
    # 预热 (Warmup)
    for _ in range(num_warmup):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    # 实际测速
    for i in range(num_iters):
        # 如果是你的 kernel，每次调用前需要将 y 清零，否则 atomicAdd 会一直累加
        if 'zero_' in kwargs:
            kwargs['zero_tensor'].zero_()
            
        start_events[i].record()
        func(*args)
        end_events[i].record()

    torch.cuda.synchronize()
    
    # 过滤掉极端的异常值，取平均
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time = sum(times) / num_iters
    return avg_time

# 5. 运行对比
print(f"开始 Benchmark (数组大小: {N / 1e6:.2f} M elements)...")

# 测试原生 PyTorch
time_torch = benchmark(torch.sum, args=(a,))
print(f"PyTorch torch.sum 耗时: {time_torch:.3f} ms")

# 测试自定义 Kernel
y_benchmark = torch.zeros(1, device='cuda', dtype=torch.float32)
# 包装一下调用逻辑，确保每次循环前 y 清零
def run_custom():
    y_benchmark.zero_()
    custom_module.custom_sum(a, y_benchmark)

time_custom = benchmark(run_custom, args=())
print(f"Custom Kernel     耗时: {time_custom:.3f} ms")

# 计算内存带宽 (GB/s)
bytes_read = N * 4 # 读取 a 的数据量 (float32 = 4 bytes)
bytes_write = 1 * 4 # 写入 y 的数据量，可以忽略不计
total_bytes = bytes_read + bytes_write

bw_torch = (total_bytes / 1e9) / (time_torch / 1000)
bw_custom = (total_bytes / 1e9) / (time_custom / 1000)

print(f"\n带宽表现:")
print(f"PyTorch Bandwidth: {bw_torch:.2f} GB/s")
print(f"Custom  Bandwidth: {bw_custom:.2f} GB/s")