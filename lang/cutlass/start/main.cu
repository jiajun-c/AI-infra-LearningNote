#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h> // CUTLASS 自带的 CPU 参考实现工具
#include <cutlass/util/tensor_view_io.h>      // 用于打印 Tensor

// 1. CPU 参考实现 (Naive GEMM)
// -----------------------------------------------------------------------------
// 为了完全掌控验证逻辑，我们手写一个简单的 CPU GEMM
// 计算公式: D = alpha * (A * B) + beta * C
template <typename TensorViewA, typename TensorViewB, typename TensorViewC, typename TensorViewD>
void cpu_gemm_reference(
    TensorViewA A, 
    TensorViewB B, 
    TensorViewC C, 
    TensorViewD D, 
    float alpha, 
    float beta) {

    int M = D.extent(0);
    int N = D.extent(1);
    int K = A.extent(1);

    // 遍历每一个输出元素 (m, n)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            
            float accum = 0.0f;
            
            // K 维点积
            for (int k = 0; k < K; ++k) {
                float a = static_cast<float>(A.at({m, k}));
                float b = static_cast<float>(B.at({k, n}));
                accum += a * b;
            }

            // Epilogue
            float c_old = static_cast<float>(C.at({m, n}));
            float d_val = alpha * accum + beta * c_old;

            D.at({m, n}) = cutlass::half_t(d_val);
        }
    }
}

int main() {
    // 2. 定义 GEMM 类型
    // -----------------------------------------------------------------------------
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,               // ElementA
        cutlass::layout::ColumnMajor,  // LayoutA
        cutlass::half_t,               // ElementB
        cutlass::layout::ColumnMajor,  // LayoutB
        cutlass::half_t,               // ElementOutput
        cutlass::layout::ColumnMajor,  // LayoutOutput
        float,                         // ElementAccumulator
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm70
    >;

    Gemm gemm_op;
    cutlass::Status status;

    // 3. 问题规模与参数
    int M = 512;
    int N = 256;
    int K = 128;

    float alpha = 1.25f;
    float beta = -1.25f;

    std::cout << "Problem Size: " << M << "x" << N << "x" << K << std::endl;

    // 4. 分配内存 (注意：这里已修正为 ColumnMajor 以匹配 Gemm 定义)
    // -----------------------------------------------------------------------------
    // 如果你坚持用 RowMajor 的 Tensor，必须修改上面的 Gemm 定义为 RowMajor
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A({M, K});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({K, N});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C({M, N}); // 原始 C (用于 beta * C)
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> D_ref({M, N}); // CPU 结果容器

    // 5. 初始化数据 (Random)
    // -----------------------------------------------------------------------------
    // 使用 host_view() 在 CPU 端填充
    for (int i = 0; i < M * K; ++i) A.host_data()[i] = static_cast<cutlass::half_t>(rand() % 5 - 2);
    for (int i = 0; i < K * N; ++i) B.host_data()[i] = static_cast<cutlass::half_t>(rand() % 5 - 2);
    for (int i = 0; i < M * N; ++i) {
        cutlass::half_t val = static_cast<cutlass::half_t>(rand() % 5 - 2);
        C.host_data()[i] = val;
        D_ref.host_data()[i] = val; // D_ref 初始化为 C 的旧值，用于 verify 逻辑
    }

    // 同步到 GPU
    A.sync_device();
    B.sync_device();
    C.sync_device();

    // 6. 启动 GPU Kernel
    // -----------------------------------------------------------------------------
    // C 作为输入源(beta * C)，也作为输出(D)
    cutlass::half_t *ptrA = A.device_data();
    cutlass::half_t *ptrB = B.device_data();
    cutlass::half_t *ptrC = C.device_data();
    cutlass::half_t *ptrD = C.device_data(); // In-place update: D writes over C

    int lda = A.device_ref().stride(0);
    int ldb = B.device_ref().stride(0);
    int ldc = C.device_ref().stride(0);
    int ldd = C.device_ref().stride(0);

    status = gemm_op({
        {M, N, K},
        {ptrA, lda},
        {ptrB, ldb},
        {ptrC, ldc},
        {ptrD, ldd},
        {alpha, beta}
    });

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM Error: " <<std::endl;
        return -1;
    }

    // 7. 同步结果回 CPU
    // -----------------------------------------------------------------------------
    C.sync_host(); // GPU 结果现在在 C.host_view() 里

    // 8. 运行 CPU 验证 (Verify)
    // -----------------------------------------------------------------------------
    std::cout << "Verifying..." << std::endl;

    // 运行 CPU 参考实现，结果存入 D_ref
    // 注意：我们将 C (原始值) 传给 cpu_gemm 的第3个参数，因为它需要读旧值
    // 但此时 C.host_view() 已经被 GPU 改写了！
    // 所以这里的验证逻辑有一个陷阱：In-place update 导致旧的 C 丢失了。
    // 为了严谨，我们通常需要 Clone 一份 C 出来，或者重新初始化 D_ref 为旧的 C。
    // *在步骤5中，我已经把旧的 C 复制了一份给 D_ref，所以这里把 D_ref 当作 Source C 传入*
    
    // 重新修正 CPU 逻辑调用：
    // D_ref (初始含旧C) = alpha * A * B + beta * D_ref (旧C)
    // 但因为 cpu_gemm_reference 是 naive 的写入，我们稍微改一下逻辑：
    // 我们用一个临时的 Tensor 来存 CPU 结果比较好，或者确保 cpu_gemm 能处理 in-place
    
    // 简单的做法：再重新生成一遍 CPU 结果
    // D_ref 目前存的是旧的 C。
    // 我们需要一个纯净的 A 和 B，以及 D_ref(作为旧C)
    
    // 为了避免逻辑混乱，我们建立一个新的 Ref Tensor
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> Reference({M, N});
    // 拷贝旧的 C (D_ref 里存的是旧 C) 到 Reference
    for(int i=0; i < M*N; ++i) Reference.host_data()[i] = D_ref.host_data()[i];

    cpu_gemm_reference(
        A.host_view(), 
        B.host_view(), 
        D_ref.host_view(), // 使用 D_ref 作为旧 C 的 Source
        Reference.host_view(), // 结果写入 Reference
        alpha, 
        beta
    );

    // 9. 逐个元素比较
    // -----------------------------------------------------------------------------
    bool passed = true;
    int errors = 0;
    float max_diff = 0.0f;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float gpu_val = static_cast<float>(C.host_view().at({m, n}));
            float cpu_val = static_cast<float>(Reference.host_view().at({m, n}));
            
            float diff = std::abs(gpu_val - cpu_val);
            if (diff > max_diff) max_diff = diff;

            // FP16 精度较低，容忍度设大一点 (e.g. 0.1)
            if (diff > 0.1f) {
                if (errors < 5) { // 只打印前5个错误
                    std::cerr << "Error at (" << m << ", " << n << "): "
                              << "GPU=" << gpu_val << ", CPU=" << cpu_val 
                              << ", Diff=" << diff << std::endl;
                }
                passed = false;
                errors++;
            }
        }
    }

    if (passed) {
        std::cout << "Verification PASSED! Max diff: " << max_diff << std::endl;
    } else {
        std::cout << "Verification FAILED! Total errors: " << errors << std::endl;
    }

    return passed ? 0 : -1;
}