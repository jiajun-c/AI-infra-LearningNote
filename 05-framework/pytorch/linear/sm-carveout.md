# PyTorch SM Carveout 机制详解

## 概述

SM Carveout 是 PyTorch 在 cuBLASLt 后端中实现的一项实验性功能，用于控制矩阵乘法运算使用的 SM（Streaming Multiprocessor）数量。这对于在多任务环境中避免资源争抢、提高整体吞吐量非常有用。

## 完整调用链路

### 1. Python API 入口

```python
# 设置 SM carveout 数量（从总 SM 数中减去的数量）
torch._C._set_sm_carveout_experimental(8)  # 例如：保留 8 个 SM 给其他任务

# 查询当前设置
torch._C._get_sm_carveout_experimental()
```

### 2. C++ 绑定层

**文件**: `torch/csrc/Module.cpp:2791-2796`

```cpp
py_module.def(
    "_set_sm_carveout_experimental", [](std::optional<int32_t> val) {
      at::globalContext()._setSMCarveout_EXPERIMENTAL(val);
    });
py_module.def("_get_sm_carveout_experimental", []() {
  return at::globalContext()._SMCarveout_EXPERIMENTAL();
});
```

### 3. Global Context 存储

**文件**: `aten/src/ATen/Context.h:403-404` 和 `aten/src/ATen/Context.cpp:626-637`

```cpp
// Context.h - 成员声明
std::optional<int32_t> _SMCarveout_EXPERIMENTAL() const;
void _setSMCarveout_EXPERIMENTAL(std::optional<int32_t> /*c*/);

// Context.cpp - 成员实现
std::optional<int32_t> Context::_SMCarveout_EXPERIMENTAL() const {
  return sm_carveout;  // 返回存储的值
}

void Context::_setSMCarveout_EXPERIMENTAL(std::optional<int32_t> c) {
  if (c.has_value()) {
    TORCH_WARN_ONCE(
      "Setting the SM carveout for matmuls is a temporary experimental mitigation for performance issues, "
      "while more robust solutions are developed. It may be removed at any moment without notice.");
  }
  sm_carveout = c;  // 存储到成员变量
}
```

**存储位置**: `Context::sm_carveout` (line 492)
```cpp
std::optional<int32_t> sm_carveout = std::nullopt;
```

### 4. cuBLASLt 调用层

**文件**: `aten/src/ATen/cuda/CUDABlas.cpp`

在多个 GEMM/BGEMM 函数中检查并应用 SM carveout 设置：

#### 4.1 bgemm_internal_cublaslt (line 468-473)

```cpp
auto stream = at::cuda::getCurrentCUDAStream();
#ifndef USE_ROCM
if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
  computeDesc.setAttribute<int32_t>(
      CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount -
          at::globalContext()._SMCarveout_EXPERIMENTAL().value());
}
#endif
```

#### 4.2 gemm_and_bias (line 1672-1677)

```cpp
auto stream = at::cuda::getCurrentCUDAStream();
#ifndef USE_ROCM
if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
  computeDesc.setAttribute<int32_t>(
      CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount -
          at::globalContext()._SMCarveout_EXPERIMENTAL().value());
}
#endif
```

#### 4.3 scaled_gemm (line 2038-2043)

```cpp
auto stream = at::cuda::getCurrentCUDAStream();
#ifndef USE_ROCM
if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
  computeDesc.setAttribute<int32_t>(
      CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount -
          at::globalContext()._SMCarveout_EXPERIMENTAL().value());
}
#endif
```

#### 4.4 int8_gemm (line 2252-2257)

```cpp
auto stream = at::cuda::getCurrentCUDAStream();
#ifndef USE_ROCM
if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
  computeDesc.setAttribute<int32_t>(
      CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount -
          at::globalContext()._SMCarveout_EXPERIMENTAL().value());
}
#endif
```

### 5. cuBLASLt API 层

**cuBLASLt 调用序列**:

```
cublasLtMatmulDescCreate()              // 创建矩阵乘法描述符
    ↓
cublasLtMatmulDescSetAttribute()        // 设置 SM_COUNT_TARGET 属性
    ↓
cublasLtMatmulAlgoGetHeuristic()        // 根据 SM 限制获取最佳算法
    ↓
cublasLtMatmul()                        // 执行矩阵乘法
```

**关键代码** (`CUDABlas.cpp:311-328`):

```cpp
class CuBlasLtMatmulDescriptor : public CuBlasLtDescriptor<
                                     cublasLtMatmulDescOpaque_t,
                                     &cublasLtMatmulDescDestroy> {
 public:
  CuBlasLtMatmulDescriptor(
      cublasComputeType_t compute_type,
      cudaDataType_t scale_type) {
    cublasLtMatmulDesc_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(
        cublasLtMatmulDescCreate(&raw_descriptor, compute_type, scale_type));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  void setAttribute(cublasLtMatmulDescAttributes_t attr, const T value) {
    TORCH_CUDABLAS_CHECK(
        ::cublasLtMatmulDescSetAttribute(descriptor(), attr, &value, sizeof(value)));
  }
};
```

## SM Carveout 计算逻辑

```
实际可用 SM 数 = GPU 总 SM 数 - SM Carveout 值

例如：
- H100 有 132 个 SM
- 设置 carveout = 8
- cuBLASLt 将限制使用 124 个 SM
```

## 应用场景

### 1. 多任务环境

当 GPU 上同时运行计算任务（如 NCCL 通信）和矩阵乘法时：

```python
# 保留 8 个 SM 给 NCCL，防止 matmul 占满所有 SM 导致通信延迟
torch._C._set_sm_carveout_experimental(8)
```

### 2. 推理服务

在多租户推理服务中，为不同请求保留资源：

```python
# 限制每个 matmul 只使用部分 SM，保证并发请求的资源隔离
torch._C._set_sm_carveout_experimental(16)
```

## 注意事项

1. **实验性功能**: SM carveout 是实验性功能，可能在未来版本中被移除或修改

2. **仅适用于 cuBLASLt**: 该功能仅在 cuBLASLt 后端生效，需要设置：
   ```python
   torch.backends.cuda.preferred_blas_library = "cublaslt"
   ```

3. **不影响实际 SM 占用**: SM carveout 只是一个**启发式提示**，cuBLASLt 会根据这个提示选择合适的算法，但**不会强制限制**实际使用的 SM 数量

4. **对 GEMV 效果有限**: 对于 GEMV（矩阵×向量）这种内存受限的操作，cuBLASLt 可能会忽略 SM 限制，因为增加 SM 并不能提升性能

5. **ROCm 实现差异**: 在 ROCm 平台上，SM carveout 通过**stream mask**实现，强制限制 SM 使用：
   ```cpp
   // ROCm 使用 hipExtStreamCreateWithCUMask 创建掩码流
   AT_CUDA_CHECK(hipExtStreamCreateWithCUMask(&stream, mask_size, &mask[0]));
   ```

## 调试和验证

### 使用 ncu 验证 SM 占用

```bash
# 查看 SM 利用率
ncu --metrics sm__cycles_active.avg.pct_of_peak_sustained_active \
    --section "SpeedOfLight" \
    ./your_program

# 查看实际 SM 周期数
ncu --metrics smsp__cycles_active.sum,sm__cycles_active.sum \
    ./your_program
```

### Python 验证

```python
import torch

# 查询 GPU 总 SM 数
props = torch.cuda.get_device_properties(0)
total_sms = props.multi_processor_count

# 设置 carveout
torch._C._set_sm_carveout_experimental(8)

# 验证设置
carveout = torch._C._get_sm_carveout_experimental()
print(f"Total SMs: {total_sms}, Carveout: {carveout}, Target: {total_sms - carveout}")
```

## 相关文件

- `torch/csrc/Module.cpp` - Python 绑定
- `aten/src/ATen/Context.h` - Context 类声明
- `aten/src/ATen/Context.cpp` - Context 类实现
- `aten/src/ATen/cuda/CUDABlas.cpp` - cuBLASLt 调用层
- `aten/src/ATen/cuda/CudaContextLight.h` - CUDA 上下文管理
- `aten/src/ATen/cuda/CublasHandlePool.cpp` - cuBLAS 句柄池管理
