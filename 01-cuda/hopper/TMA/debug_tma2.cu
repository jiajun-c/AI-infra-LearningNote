#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

using namespace cute;

// TMA kernel - 使用静态形状
__global__ void tma_hello_world(auto tma_load, float* d_out) {
    // 使用 CuTe 的 SMEM layout（column-major，即默认）
    __shared__ float smem[16 * 16];
    Tensor sA = make_tensor(make_smem_ptr(smem), make_layout(make_shape(_16{}, _16{})));

    __shared__ alignas(8) uint64_t tma_barrier;

    if (threadIdx.x == 0) {
        cute::initialize_barrier(tma_barrier, 1);
        uint32_t tx_bytes = 16 * 16 * sizeof(float);
        cute::set_barrier_transaction_bytes(tma_barrier, tx_bytes);

        // get_tma_tensor 的参数应该是全局 tensor 的形状（和 host 端一致）
        Tensor gA = tma_load.get_tma_tensor(make_shape(32, 32));
        
        auto cta_tma = tma_load.get_slice(Int<0>{});
        // partition_S 对全局 tensor 做分区
        Tensor tXgA = cta_tma.partition_S(gA);
        // partition_D 对 smem tensor 做分区
        Tensor tXsA = cta_tma.partition_D(sA);

        // 打印分区结果
        print("tXgA shape: "); print(tXgA.shape()); print("\n");
        print("tXsA shape: "); print(tXsA.shape()); print("\n");
        print("tXgA layout: "); print(tXgA.layout()); print("\n");

        // 选取第 0 个 tile
        cute::copy(tma_load.with(tma_barrier), tXgA(_, 0, 0), tXsA(_, 0, 0));
        cute::wait_barrier(tma_barrier, 0);

        printf("Thread 0 after TMA: smem[0..4] = %f %f %f %f %f\n",
               smem[0], smem[1], smem[2], smem[3], smem[4]);
    }

    __syncthreads();

    int idx = threadIdx.x;
    if (idx < 256) {
        d_out[idx] = smem[idx];
    }
}

int main() {
    int M = 32, N = 32;
    int tile_M = 16, tile_N = 16;

    std::vector<float> h_in(M * N);
    for (int i = 0; i < M * N; ++i) h_in[i] = static_cast<float>(i);

    float *d_in, *d_out;
    cudaMalloc(&d_in, M * N * sizeof(float));
    cudaMalloc(&d_out, tile_M * tile_N * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    // 全局 tensor - column major (CuTe 默认)
    auto layout_g = make_layout(make_shape(M, N));   // stride = (1, 32)  => column major
    auto layout_s = make_layout(make_shape(_16{}, _16{}));  // stride = (1, 16) => column major
    Tensor gA = make_tensor(d_in, layout_g);

    auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gA, layout_s);

    print("Host - gA layout: "); print(gA.layout()); print("\n");
    print("Host - sA layout: "); print(layout_s); print("\n");

    tma_hello_world<<<1, 256>>>(tma_load, d_out);
    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::vector<float> h_out(tile_M * tile_N);
    cudaMemcpy(h_out.data(), d_out, tile_M * tile_N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "TMA output first 16 elements (row 0 of smem, column-major):" << std::endl;
    for (int i = 0; i < 16; i++) std::cout << h_out[i] << " ";
    std::cout << std::endl;

    std::cout << "Expected (column-major, first 16 of col 0): 0 1 2 3 ... 15" << std::endl;

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
