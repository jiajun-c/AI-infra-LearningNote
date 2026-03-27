#include <torch/extension.h>
#include "vector_add_kernel.cuh"

// ============================================================
// pybind11 方式绑定自定义算子
// ============================================================
// 问题1: 对 torch.compile 来说是"黑盒" → 导致 graph break
// 问题2: 使用默认 stream(0) → CUDA Graph capture 时不被捕获
// 问题3: PyTorch Python 层不会报错 → 静默产生错误结果

torch::Tensor pybind_vector_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "输入必须是CUDA tensor");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "输入必须是contiguous");
    TORCH_CHECK(a.sizes() == b.sizes(), "输入shape必须一致");

    auto c = torch::empty_like(a);
    int N = a.numel();

    // ⚠️ 关键问题: 这里使用默认 stream (stream 0)
    // CUDA Graph capture 期间，PyTorch 使用的是 capture stream (非默认stream)
    // 但 pybind11 函数对此一无所知，仍在默认 stream 上启动 kernel
    // 结果: kernel 实际执行了，但没有被 graph 捕获
    //       graph replay 时不会重放这个 kernel → 结果错误
    launch_vector_add(a.data_ptr<float>(), b.data_ptr<float>(),
                      c.data_ptr<float>(), N);
    // 注意: 这里没有传 stream 参数，默认使用 stream 0

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &pybind_vector_add,
          "Vector add via pybind11 (CUDA Graph 不兼容!)");
}
