#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

using namespace cute;

// 简单不使用 TMA 的 kernel 来验证输入数据是否正确
__global__ void simple_copy(float* d_in, float* d_out, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) d_out[idx] = d_in[idx];
}

// TMA kernel
__global__ void tma_hello_world(auto tma_load, float* d_out) {
    __shared__ float smem[16 * 16];
    Tensor sA = make_tensor(make_smem_ptr(smem), make_layout(make_shape(_16{}, _16{})));

    __shared__ alignas(8) uint64_t tma_barrier;

    if (threadIdx.x == 0) {
        cute::initialize_barrier(tma_barrier, 1);
        uint32_t tx_bytes = 16 * 16 * sizeof(float);
        cute::set_barrier_transaction_bytes(tma_barrier, tx_bytes);

        Tensor gA = tma_load.get_tma_tensor(make_shape(_32{}, _32{}));
        Tensor gA_tile = local_tile(gA, make_shape(_16{}, _16{}), make_coord(0, 0));

        auto cta_tma = tma_load.get_slice(Int<0>{});
        Tensor tXgA = cta_tma.partition_S(gA_tile);
        Tensor tXsA = cta_tma.partition_D(sA);

        // Print partition shapes
        if (threadIdx.x == 0) {
            print("gA shape: "); print(gA.shape()); print("\n");
            print("gA_tile shape: "); print(gA_tile.shape()); print("\n");
            print("tXgA shape: "); print(tXgA.shape()); print("\n");
            print("tXsA shape: "); print(tXsA.shape()); print("\n");
        }

        cute::copy(tma_load.with(tma_barrier), tXgA, tXsA);
        cute::wait_barrier(tma_barrier, 0);

        // Print first values from smem right after TMA completes
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

    // 1. First verify input data is correct
    std::cout << "=== Step 1: Verify input data ===" << std::endl;
    float *d_verify;
    cudaMalloc(&d_verify, 5 * sizeof(float));
    simple_copy<<<1, 5>>>(d_in, d_verify, 5);
    cudaDeviceSynchronize();
    float h_verify[5];
    cudaMemcpy(h_verify, d_verify, 5*sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Input d_in[0..4]: ";
    for (int i = 0; i < 5; i++) std::cout << h_verify[i] << " ";
    std::cout << std::endl;
    cudaFree(d_verify);

    // 2. TMA test
    std::cout << "\n=== Step 2: TMA test ===" << std::endl;
    auto layout_g = make_layout(make_shape(M, N));
    auto layout_s = make_layout(make_shape(_16{}, _16{}));
    Tensor gA = make_tensor(d_in, layout_g);
    auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gA, layout_s);

    // Print TMA info on host
    print("Host - TMA layout_g: "); print(layout_g); print("\n");
    print("Host - TMA layout_s: "); print(layout_s); print("\n");

    tma_hello_world<<<1, 256>>>(tma_load, d_out);
    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::vector<float> h_out(tile_M * tile_N);
    cudaMemcpy(h_out.data(), d_out, tile_M * tile_N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "TMA output[0..15]: ";
    for (int i = 0; i < 16; i++) std::cout << h_out[i] << " ";
    std::cout << std::endl;

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
