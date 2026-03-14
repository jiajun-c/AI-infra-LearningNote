#include <iostream>
#include <vector>
#include <cute/tensor.hpp>

using namespace cute;

using TA = float;
constexpr int M = 16;
constexpr int N = 8; // 16x8 = 128 elements

// ==============================================================================
// 异步拷贝 Kernel: 从 Global Memory 异步搬运到 Shared Memory
// ==============================================================================
__global__ void async_copy_kernel(const TA* g_in, TA* g_out) {
    __shared__ TA smem[M * N];
    int tid = threadIdx.x;

    // 1. 包装张量 (行优先)
    Tensor g_in_tensor  = make_tensor(make_gmem_ptr(g_in),  make_layout(Shape<_16, _8>{}, LayoutRight{}));
    Tensor s_tensor     = make_tensor(make_smem_ptr(smem),  make_layout(Shape<_16, _8>{}, LayoutRight{}));
    Tensor g_out_tensor = make_tensor(make_gmem_ptr(g_out), make_layout(Shape<_16, _8>{}, LayoutRight{}));

    // 2. 构建异步拷贝蓝图 (Global -> Shared)
    // 关键点：使用 SM80_CP_ASYNC_CACHEGLOBAL 指令，每次拷贝 uint128_t (16 bytes = 4 个 float)
// 完美的 TiledCopy 蓝图：契合 Row-Major (16x8)
    auto copy_async = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, TA>{}, 
        
        // 【线程布局 ThreadLayout】
        // 32 个线程排成 16 行 2 列，且行优先 (Stride<_2, _1>)
        // 这样线程 0,1 负责第 0 行；线程 2,3 负责第 1 行...
        Layout<Shape<_16, _2>, Stride<_2, _1>>{}, 
        
        // 【值布局 ValueLayout】
        // 每个线程在 N 维度 (连续维度) 上抓取 4 个元素
        Layout<Shape<_1, _4>>{}   
    );

    auto thr_copy = copy_async.get_thread_slice(tid);

    // 3. 切分数据源 (Source: Global)
    Tensor tAgA_src = thr_copy.partition_S(g_in_tensor);


    // =======================================================================
    // 💥💥💥 引爆开关：请在这里切换注释！ 💥💥💥
    // =======================================================================

    // ✅ 正确示范：使用 _D 切分目的地 (Shared)
    Tensor tAsA_dst = thr_copy.partition_S(s_tensor);

    // ❌ 错误示范：使用 _S 切分目的地 (Shared)。解开这行注释，注释掉上面那行，去编译试试！
    // Tensor tAsA_dst = thr_copy.partition_S(s_tensor);

    // =======================================================================


    // 4. 触发拷贝 (带蓝图的 copy)
    cute::copy(copy_async, tAgA_src, tAsA_dst);
    
    // 5. 提交并等待异步拷贝完成
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // 6. 验证：用普通的同步拷贝把 Shared Memory 写回 Global Memory 的 out 中
    auto copy_sync = make_tiled_copy(Copy_Atom<DefaultCopy, TA>{}, Layout<Shape<_32, _1>>{}, Layout<Shape<_4, _1>>{});
    auto thr_copy_sync = copy_sync.get_thread_slice(tid);
    
    // 这里的写回操作，严谨地遵循了 源用S，目标用D 的规矩
    Tensor tAsA_src2 = thr_copy_sync.partition_D(s_tensor);
    Tensor tAgA_dst2 = thr_copy_sync.partition_D(g_out_tensor);
    cute::copy(copy_sync, tAsA_src2, tAgA_dst2);
}

int main() {
    int num_elements = M * N;
    size_t bytes = num_elements * sizeof(TA);

    std::vector<TA> h_in(num_elements);
    std::vector<TA> h_out(num_elements, 0.0f);
    for (int i = 0; i < num_elements; ++i) h_in[i] = static_cast<TA>(i);

    TA *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    std::cout << "🚀 Launching Async CuTe Kernel..." << std::endl;
    async_copy_kernel<<<1, 32>>>(d_in, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    // 简单验证
    bool success = true;
    for (int i = 0; i < num_elements; ++i) {
        if (h_in[i] != h_out[i]) {
            std::cerr << "Mismatch at index " << i << std::endl;
            success = false;
            break;
        }
    }
    if (success) std::cout << "✅ Async Copy execution verified successfully!" << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}