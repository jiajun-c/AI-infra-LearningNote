#include "cute/layout.hpp"
#include "cute/layout_composed.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/pointer.hpp"
#include "cute/pointer_flagged.hpp"
#include "cute/stride.hpp"
#include "cute/tensor_impl.hpp"
#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

__global__ void copy_kernel(float* d_in, float* d_out, int M, int N) {
    auto global_layout = make_layout(make_shape(M, N), LayoutLeft{});

    Tensor g_in_full = make_tensor(make_gmem_ptr(d_in), global_layout);
    Tensor g_out_full = make_tensor(make_gmem_ptr(d_out), global_layout);
    auto tiler = Shape<_4, _8>{};
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < M*N/(4*8)) {
        auto tile_in = local_tile(g_in_full, tiler, tid);
        auto tile_out = local_tile(g_out_full, tiler, tid);
        copy(tile_in, tile_out);
    }
}

int main() {
    int M = 128, N = 128;
    float *h_data_in = new float[M*N];
    float *d_data_in, *d_data_out;
    for (int i = 0; i < M*N; i++) h_data_in[i] = i;

    cudaMalloc((void**)&d_data_in, M*N*sizeof(float));
    cudaMemcpy(d_data_in, h_data_in, M*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_data_out, M*N*sizeof(float));
    dim3 block = dim3{128};
    dim3 grid = dim3{4};
    copy_kernel<<<grid, block>>>(d_data_in, d_data_out,  M,  N);
    cudaDeviceSynchronize();
    float *h_data_out = new float[M*N];
    cudaMemcpy(h_data_out, d_data_out, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; i++) {
        printf("%f ", h_data_out[i*128]);
    }
    // 1. 定义一个 Layout
    // 形状: (4, 8)
    // 步长: (1, 4) -> Column-Major (列主序)，即 LayoutLeft
    // auto layout = make_layout(make_shape(8, 24), LayoutLeft{});
    // auto tiler = Shape<_4, _8>{};
    // int *data = new int[8*24];
    // for (int i = 0; i < 8*24; i++) data[i] = i;
    // Tensor a = make_tensor(data, layout);
    // Tensor tile_a = zipped_divide(a, tiler);
    // // 2. 打印 Layout 信息
    // auto tile_b = local_tile(a, tiler, make_coord(1, 0));

    // print(tile_b(0, 0));print("\n");
    // auto element= tile_a(make_coord(0, 1), make_coord(0, 0));
    // print(element);
    // return 0;
}

// (2, 3)
// (3, 4);

// 3*24 + 4 = 72 + 4 = 76