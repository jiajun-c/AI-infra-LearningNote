#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#define CUTE_CHECK_LAST() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

// 修正点1: 强制 Shared Memory 对齐，防止 cp.async 报错
// 使用 union 技巧确保指针是 128-bit (16 Byte) 对齐的
template<typename T>
struct alignas(16) AlignedStorage {
    T data;
};

template <class TiledCopy>
__global__ void test_async_copy_kernel(float const* d_in, float* d_out, int N, TiledCopy copy_op) {
    using namespace cute;

    // 我们将数据指针强转为 uint128_t* 来看待
    // 这样 1 个元素 = 4 个 float
    using GlobalType = uint128_t; 

    // 1. 定义 Tensor
    // 注意：这里 shape 是 (N / 4)，因为我们现在的视角是 128-bit 宽度的元素
    Tensor mIn  = make_tensor(make_gmem_ptr((GlobalType const*)d_in),  make_shape(N / 4));
    Tensor mOut = make_tensor(make_gmem_ptr((GlobalType*)d_out), make_shape(N / 4));

    // 计算当前 Block 负责的 128-bit 元素个数 (128 个线程 * 1 = 128 个 uint128_t)
    auto elems_128b_per_block = size(TiledCopy{}); 
    
    int bid = blockIdx.x;
    // 越界保护
    if (bid * elems_128b_per_block >= 1024*1024) return;

    // 2. 切分 Global Tile
    Tensor gIn_tile  = local_tile(mIn,  make_shape(elems_128b_per_block), make_coord(bid));
    Tensor gOut_tile = local_tile(mOut, make_shape(elems_128b_per_block), make_coord(bid));

    // 3. 定义 Shared Memory
    // 修正点2: 必须确保容量足够且对齐
    extern __shared__ uint128_t smem_buffer[]; 
    Tensor sMem = make_tensor(make_smem_ptr(smem_buffer), make_shape(elems_128b_per_block));

    // 4. 线程划分
    auto loader = copy_op.get_slice(threadIdx.x);
    Tensor tIg = loader.partition_S(gIn_tile);
    Tensor tIs = loader.partition_D(sMem);

    // =========================================================
    // 异步拷贝
    // =========================================================
    
    // 发起拷贝: Global (uint128_t) -> Shared (uint128_t)
    copy(copy_op, tIg, tIs); 

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads(); 

    // =========================================================
    // 写回 Global (用于验证)
    // =========================================================
    
    Tensor tOs = loader.partition_S(sMem);      
    Tensor tOg = loader.partition_D(gOut_tile); 

    copy(tOs, tOg); 
}

int main() {
    using namespace cute;

    // 检查架构
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    if (props.major < 8) {
        std::cout << "Need Ampere (SM80)+ GPU" << std::endl;
        return 0;
    }

    int N = 1024 * 1024; // 1M floats
    // 确保 N 是 4 的倍数 (128-bit 对齐)
    assert(N % 4 == 0);

    // 策略定义
    using BlockThreads = Int<128>;
    
    // 修正点3: 明确 CopyOp 的语义
    // Atom: 使用 uint128_t 进行拷贝
    // ValLayout: 每个线程处理 1 个 uint128_t (即 1 个 128-bit 向量)
    using CopyOp = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, uint128_t>{}, 
        Layout<Shape<BlockThreads>>{}, // Thread Layout: 128
        Layout<Shape<_1>>{}            // Val Layout: 1 个 uint128_t
    ));
    
    int tile_size_128b = size(CopyOp{}); // 应该是 128
    int tile_size_floats = tile_size_128b * 4; // 真实 float 数量 = 512
    
    // 修正点4: 计算字节数时使用 16 字节
    size_t smem_size_bytes = tile_size_128b * sizeof(uint128_t);

    std::cout << "Threads per Block: " << size(BlockThreads{}) << std::endl;
    std::cout << "Elements(128b) per Tile: " << tile_size_128b << std::endl;
    std::cout << "Elements(float) per Tile: " << tile_size_floats << std::endl;
    std::cout << "Smem per Block: " << smem_size_bytes << " bytes" << std::endl;

    // 准备数据
    thrust::host_vector<float> h_in(N);
    for(int i=0; i<N; ++i) h_in[i] = static_cast<float>(i);
    
    thrust::device_vector<float> d_in = h_in;
    thrust::device_vector<float> d_out(N, -1.0f);

    int num_blocks = (N / 4 + tile_size_128b - 1) / tile_size_128b;
    
    test_async_copy_kernel<<<num_blocks, 128, smem_size_bytes>>>(
        thrust::raw_pointer_cast(d_in.data()),
        thrust::raw_pointer_cast(d_out.data()),
        N,
        CopyOp{}
    );
    CUTE_CHECK_LAST();
    cudaDeviceSynchronize();

    // 验证
    thrust::host_vector<float> h_out = d_out;
    bool correct = true;
    for(int i=0; i<N; ++i) {
        if (h_out[i] != h_in[i]) {
            std::cout << "Mismatch at " << i << ": expected " << h_in[i] 
                      << ", got " << h_out[i] << std::endl;
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "Success! cp.async vector copy works." << std::endl;
    }

    return 0;
}