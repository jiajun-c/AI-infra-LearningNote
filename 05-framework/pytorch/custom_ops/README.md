# 自定义 CUDA 算子与 CUDA Graph 兼容性

## 背景

当我们编写自定义 CUDA 算子并集成到 PyTorch 中时，**绑定方式的选择**直接影响：
1. `torch.compile` 是否产生 graph break
2. CUDA Graph 能否正确捕获和重放
3. **最危险的是：pybind11 方式在 CUDA Graph 下可能静默产生错误结果，Python 层不会报错**

本示例通过同一个 CUDA kernel，对比两种绑定方式的行为差异。

## 两种绑定方式

### 方式 1: pybind11 (`PYBIND11_MODULE`)

直接将 C++ 函数暴露为 Python 函数：

```cpp
torch::Tensor pybind_vector_add(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    launch_vector_add(a.data_ptr<float>(), b.data_ptr<float>(),
                      c.data_ptr<float>(), a.numel());
    // ⚠️ 使用默认 stream (stream 0)
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &pybind_vector_add);
}
```

**问题**：对 PyTorch 调度器来说是"黑盒"，不参与 stream 管理、graph tracing。

### 方式 2: `TORCH_LIBRARY` + `TORCH_LIBRARY_IMPL`

注册到 PyTorch 算子调度器：

```cpp
torch::Tensor torchlib_vector_add(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();  // ✅ 关键!
    launch_vector_add(a.data_ptr<float>(), b.data_ptr<float>(),
                      c.data_ptr<float>(), a.numel(), stream);
    return c;
}

TORCH_LIBRARY(custom_ops, m) {
    m.def("vector_add(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_ops, CUDA, m) {
    m.impl("vector_add", torchlib_vector_add);
}

TORCH_LIBRARY_IMPL(custom_ops, Meta, m) {
    m.impl("vector_add", vector_add_meta);  // shape inference
}
```

**优势**：完全集成到 PyTorch 调度器，stream/graph/compile 全部兼容。

## 核心问题：Stream 不匹配

```
正常执行 (Eager):
  PyTorch 默认 stream ──→ pybind11 用 stream 0 ──→ 同一个 stream, 没问题

CUDA Graph Capture:
  PyTorch 切换到 capture stream ──→ pybind11 仍用 stream 0
                                     ↓
                          kernel 在 stream 0 上执行
                          但 graph 只捕获 capture stream 上的操作
                                     ↓
                          kernel 未被捕获! Replay 时不会重放!

TORCH_LIBRARY_IMPL:
  at::cuda::getCurrentCUDAStream() 返回 capture stream
  → kernel 在 capture stream 上执行 → 被 graph 正确捕获
```

## 实验结果

### 实验 1: Eager 正确性

两种方式在 Eager 模式下都能正确计算，**问题在 Eager 下完全隐藏**。

### 实验 2: torch.compile Graph Break

```
pybind11:       Graph 数量 = 2, Graph Break 数量 = 1
  Break 原因: unsupported builtin pybind_vector_add.PyCapsule.vector_add
TORCH_LIBRARY:  Graph 数量 = 1, Graph Break 数量 = 0
```

pybind11 导致 graph break，`torch.compile` 无法将自定义算子纳入计算图优化。

### 实验 3: CUDA Graph 兼容性 (核心!)

```
--- pybind11 + CUDA Graph ---
  Python层报错:   没有!
  Replay结果正确: False
  期望值 (全3.0): [3.0, 3.0, 3.0, 3.0, 3.0]
  实际值:          [1.42, -2.86, 1.45, 0.12, 2.80]   ← 错误!
  ⚠️ kernel 没有被 CUDA Graph 捕获，但 Python 层不报错!

--- TORCH_LIBRARY_IMPL + CUDA Graph ---
  Replay结果正确: True
  期望值 (全3.0): [3.0, 3.0, 3.0, 3.0, 3.0]
  实际值:          [3.0, 3.0, 3.0, 3.0, 3.0]         ← 正确
```

### 实验 4: 性能对比

| 模式 | pybind11 | TORCH_LIBRARY_IMPL |
|------|----------|---------------------|
| Eager | 0.034 ms | 0.045 ms |
| torch.compile | 0.035 ms (有 break) | 0.063 ms (无 break) |
| CUDA Graph | N/A (结果错误) | **0.031 ms** (最快) |

CUDA Graph 消除了 kernel launch 开销，是性能最优的执行方式，但**仅 TORCH_LIBRARY_IMPL 方式兼容**。

## 总结对比

| 特性 | pybind11 | TORCH_LIBRARY_IMPL |
|------|----------|---------------------|
| Eager 模式 | ✅ 正确 | ✅ 正确 |
| torch.compile | ❌ Graph Break | ✅ 无 Break |
| CUDA Graph Capture | ❌ 静默失败 | ✅ 正确捕获 |
| CUDA Graph Replay | ❌ 结果错误 | ✅ 结果正确 |
| Python 层报错 | ❌ 不报错(危险!) | ✅ N/A |
| Meta/FakeTensor | ❌ 不支持 | ✅ 支持 |
| Dispatcher 集成 | ❌ 不集成 | ✅ 完全集成 |

## 最佳实践

1. **新项目**: 始终使用 `TORCH_LIBRARY_IMPL` 或 `torch.library.custom_op`
2. **已有 pybind11 代码**: 至少使用 `at::cuda::getCurrentCUDAStream()` 获取正确 stream
3. **需要 torch.compile**: 必须提供 Meta 实现做 shape inference
4. **需要 CUDA Graph**: 必须正确处理 stream，否则会静默失败

## 运行

```bash
python3 compare.py
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `vector_add_kernel.cuh` | 共享 CUDA kernel（两种方式共用） |
| `pybind_vector_add.cu` | pybind11 绑定（使用默认 stream → 不兼容 CUDA Graph） |
| `torchlib_vector_add.cu` | TORCH_LIBRARY_IMPL 绑定（使用 getCurrentCUDAStream → 兼容） |
| `compare.py` | Python 对比脚本（一键运行所有实验） |

## 相关内容

- [CUDA Graph 基础](../graph/README.md)
- [torch.compile + Triton 自定义算子](../compile/)
- [pybind11 基础](../../../08-tools/glue/torch/pybind/)

## 参考

- [PyTorch Custom Operators 官方指南](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
- [TORCH_LIBRARY API](https://pytorch.org/docs/stable/library.html)
- [CUDA Graphs 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
