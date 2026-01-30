# 使用cutlass进行矩阵乘法运算

## 1. 数据存储

cutlass中可以使用一个Tensor类进行存储，通过sync_device和sync_host的形式在device和host之间进行数据的传递


## 2. 计算

如下所示，先在host端创建元素，然后sync_device同步到host_device，再通过gemm的接口进行计算

```cpp
    cutlass::HostTensor<cutlass::half_t, Layout>tensor_A({M, K});
    cutlass::HostTensor<cutlass::half_t, Layout>tensor_B({K, N});
    cutlass::HostTensor<cutlass::half_t, Layout>tensor_C({M, N});

    for (int i = 0; i < tensor_A.capacity(); ++i) tensor_A.host_data()[i] = 1.0_hf;
    for (int i = 0; i < tensor_B.capacity(); ++i) tensor_B.host_data()[i] = 1.0_hf;
    for (int i = 0; i < tensor_C.capacity(); ++i) tensor_C.host_data()[i] = 0.0_hf;

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();

    Gemm gemm_op;
    
    cutlass::Status status = gemm_op({
        {M, N, K},                  // Problem Size
        {tensor_A.device_data(), tensor_A.stride(0)},  // A (ptr, stride)
        {tensor_B.device_data(), tensor_B.stride(0)},  // B
        {tensor_C.device_data(), tensor_C.stride(0)},  // C
        {tensor_C.device_data(), tensor_C.stride(0)},  // D (Output)
        {alpha, beta}               // Scalars
    });
```