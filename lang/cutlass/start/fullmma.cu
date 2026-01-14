#include <iostream>
#include <iomanip>
#include <cstdlib> // for rand()

// CUTLASS 核心头文件
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h> // 关键工具：HostTensor

// 定义打印矩阵的辅助函数
template <typename TensorView>
void print_tensor_view(TensorView view, const char* name, int rows_to_print = 6, int cols_to_print = 6) {
    std::cout << "Matrix " << name << " (Top-left " << rows_to_print << "x" << cols_to_print << "):" << std::endl;
    
    // 获取实际维度
    int rows = view.extent(0);
    int cols = view.extent(1);

    for (int r = 0; r < std::min(rows, rows_to_print); ++r) {
        for (int c = 0; c < std::min(cols, cols_to_print); ++c) {
            // cutlass::half_t 需要先转为 float 才能用 cout 打印
            float val = static_cast<float>(view.at({r, c}));
            std::cout << std::setw(8) << std::setprecision(4) << val << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    // 1. 定义 GEMM 类型 (保持你的配置不变)
    // ----------------------------------------------------------------
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,               // ElementA
        cutlass::layout::ColumnMajor,  // LayoutA
        cutlass::half_t,               // ElementB
        cutlass::layout::ColumnMajor,  // LayoutB
        cutlass::half_t,               // ElementOutput
        cutlass::layout::ColumnMajor,  // LayoutOutput
        float,                         // ElementAccumulator
        cutlass::arch::OpClassTensorOp,// tag indicating Tensor Cores
        cutlass::arch::Sm70            // tag: Volta (Sm70), 如果你是3090/4090也可以填Sm80
    >;

    Gemm gemm_op;
    cutlass::Status status;

    // 2. 定义问题规模
    // ----------------------------------------------------------------
    int M = 512;
    int N = 256;
    int K = 128;

    float alpha = 1.0f; // 设为 1.0 方便观察结果 (C = A*B)
    float beta = 0.0f;  // 设为 0.0 不累加 C 的旧值

    std::cout << "Problem Size: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // 3. 分配内存 (HostTensor 自动管理 Host 和 Device 内存)
    // ----------------------------------------------------------------
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A({M, K});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({K, N});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C({M, N});

    // 4. 数据填充 (在 Host 端进行)
    // ----------------------------------------------------------------
    // 为了方便验证，我们填充简单的规律数据
    // A 填充为 1.0, B 填充为 1.0 -> 结果 C 应该全为 K (128.0)
    
    // 获取 Host 端的视图 (TensorView) 进行操作
    auto host_A = A.host_view();
    auto host_B = B.host_view();
    auto host_C = C.host_view();

    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < K; ++c) {
            // 这里我们用随机数或者固定值，这里用随机数模拟真实情况
            // 为了防止溢出，生成小一点的数
            float val = static_cast<float>(rand() % 5) - 2.0f; 
            host_A.at({r, c}) = cutlass::half_t(val);
        }
    }

    for (int r = 0; r < K; ++r) {
        for (int c = 0; c < N; ++c) {
            float val = static_cast<float>(rand() % 5) - 2.0f;
            host_B.at({r, c}) = cutlass::half_t(val);
        }
    }

    // 初始化 C 为 0
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            host_C.at({r, c}) = cutlass::half_t(0.0f);
        }
    }

    // 打印输入矩阵的一部分看看
    print_tensor_view(host_A, "A (Input)");
    print_tensor_view(host_B, "B (Input)");

    // 5. 同步数据 Host -> Device
    // ----------------------------------------------------------------
    // 这一步至关重要！否则 GPU 上读到的全是 0 或垃圾数据
    A.sync_device();
    B.sync_device();
    C.sync_device(); 

    // 6. 准备参数并启动 Kernel
    // ----------------------------------------------------------------
    // 使用 .device_data() 获取 GPU 指针
    cutlass::half_t const *ptrA = A.device_data();
    cutlass::half_t const *ptrB = B.device_data();
    cutlass::half_t const *ptrC = C.device_data();
    cutlass::half_t       *ptrD = C.device_data();

    // 使用 .device_ref() 获取步长 (Stride)
    // 对于 ColumnMajor, stride(0) 通常是 Leading Dimension (LD) 即行数
    int lda = A.device_ref().stride(0);
    int ldb = B.device_ref().stride(0);
    int ldc = C.device_ref().stride(0);
    int ldd = C.device_ref().stride(0);

    std::cout << "Launching CUTLASS GEMM..." << std::endl;

    status = gemm_op({
        {M, N, K},
        {ptrA, lda},      // TensorRef A
        {ptrB, ldb},      // TensorRef B
        {ptrC, ldc},      // TensorRef C (Source accumulator)
        {ptrD, ldd},      // TensorRef D (Destination)
        {alpha, beta}     // Epilogue: D = alpha * A*B + beta * C
    });

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM failed: "  << std::endl;
        return -1;
    }

    // 7. 同步数据 Device -> Host 并打印结果
    // ----------------------------------------------------------------
    // 必须同步回来，否则 host_view 里的数据还是旧的
    C.sync_host();

    std::cout << "Computation Success." << std::endl;
    print_tensor_view(C.host_view(), "C (Result)");

    return 0;
}