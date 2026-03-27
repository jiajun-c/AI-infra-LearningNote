#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "vector_add_kernel.cuh"

// ============================================================
// TORCH_LIBRARY_IMPL 方式绑定自定义算子
// ============================================================
// 优势1: 注册到 PyTorch 调度器 → torch.compile 可见，不 graph break
// 优势2: 使用 at::cuda::getCurrentCUDAStream() → CUDA Graph 兼容
// 优势3: 提供 Meta 实现 → torch.compile 编译期可推导 shape

// CUDA 后端的实际实现
torch::Tensor torchlib_vector_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "输入必须是CUDA tensor");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "输入必须是contiguous");
    TORCH_CHECK(a.sizes() == b.sizes(), "输入shape必须一致");

    auto c = torch::empty_like(a);
    int N = a.numel();

    // ✅ 关键区别: 使用 PyTorch 当前的 CUDA stream
    // 正常执行时: 返回默认 stream → 正常工作
    // CUDA Graph capture 时: 返回 capture stream → kernel 被正确捕获到 graph 中
    // Graph replay 时: 重放捕获的 kernel → 结果正确
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_vector_add(a.data_ptr<float>(), b.data_ptr<float>(),
                      c.data_ptr<float>(), N, stream);

    return c;
}

// Meta/FakeTensor 实现 (torch.compile 编译期需要)
// 只做 shape inference，不执行实际计算
// 没有这个实现，torch.compile 在 tracing 阶段就会失败
torch::Tensor vector_add_meta(torch::Tensor a, torch::Tensor b) {
    return torch::empty_like(a);
}

// 1. 声明算子 schema (定义算子接口)
TORCH_LIBRARY(custom_ops, m) {
    m.def("vector_add(Tensor a, Tensor b) -> Tensor");
}

// 2. 为 CUDA 后端注册实现
TORCH_LIBRARY_IMPL(custom_ops, CUDA, m) {
    m.impl("vector_add", torchlib_vector_add);
}

// 3. 注册 Meta 实现 (torch.compile 需要做 shape inference)
TORCH_LIBRARY_IMPL(custom_ops, Meta, m) {
    m.impl("vector_add", vector_add_meta);
}