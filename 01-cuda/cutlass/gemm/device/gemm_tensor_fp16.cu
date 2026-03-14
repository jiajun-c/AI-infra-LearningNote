#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/host_tensor.h"

int main() {
    int M = 128, N = 128, K = 128;
    cutlass::half_t alpha = 1.0_hf;
    cutlass::half_t beta = 1.0_hf;

    using Layout = cutlass::layout::ColumnMajor;
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, Layout,  // Matrix A
        cutlass::half_t, Layout,  // Matrix B
        cutlass::half_t, Layout   // Matrix C
    >;

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

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM failed: " <<  std::endl;
        return -1;
    }

    // 同步
    cudaDeviceSynchronize();

    // 7. 验证结果
    // 将结果从 Device 同步回 Host
    tensor_C.sync_host();

    // 打印第一个元素验证 (应该等于 K = 128)
    std::cout << "C[0] = " << float(tensor_C.host_data()[0]) << std::endl;
    std::cout << "CUTLASS FP16 GEMM executed successfully!" << std::endl;

    return 0;
}