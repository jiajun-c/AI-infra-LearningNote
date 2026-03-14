#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

// ============================================================================
// 核心算子声明 - 由用户实现
// ============================================================================
// TODO: 实现你的矩阵转置核函数
// 核心算子实现
template <typename T, int kM, int kN>
__global__ void cute_tiled_transpose_kernel(T const* S_ptr, T* D_ptr) {
    using namespace cute;

    // 1. 定义全局布局 (保持不变)
    auto layout_S = make_layout(make_shape(Int<kM>{}, Int<kN>{}), GenRowMajor{});
    auto layout_D = make_layout(make_shape(Int<kN>{}, Int<kM>{}), GenRowMajor{}); 

    Tensor gS = make_tensor(make_gmem_ptr(S_ptr), layout_S);
    Tensor gD = make_tensor(make_gmem_ptr(D_ptr), layout_D);

    // 2. Tile 定义
    auto bM = Int<32>{};
    auto bN = Int<32>{};
    auto bS = make_shape(bM, bN);
    auto swizzle = Swizzle<3, 3, 3>{};

    // 3. Shared Memory 与 Swizzle (保持不变)
    auto smem_layout_S = composition(swizzle, make_layout(make_shape(bM, bN), make_stride(bN, Int<1>{})));
    // 读出视图: Col-Major + Swizzle (逻辑转置)
    auto smem_layout_D = composition(swizzle, make_layout(make_shape(bM, bN), make_stride(Int<1>{}, bN)));
    __shared__ T smem_data[size(bS)];
    Tensor sS = make_tensor(make_smem_ptr(smem_data), smem_layout_S);
    Tensor sD = make_tensor(make_smem_ptr(smem_data), smem_layout_D);

    // 4. 定义 Tiled Copy
    // 我们定义一个 32 线程的拷贝器 (8x4 线程排列)，每个线程负责 4 个连续元素 (1x4)
    using CopyInst = Copy_Atom<DefaultCopy, T>; 
    auto tiled_copy = make_tiled_copy(
        CopyInst{},
        make_layout(make_shape(Int<8>{}, Int<4>{}), GenRowMajor{}), 
        make_layout(make_shape(Int<1>{}, Int<4>{}))                
    );
    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    // 5. 计算 Block 区域
    auto c_tile = make_coord(blockIdx.x, blockIdx.y);
    Tensor gS_tile = local_tile(gS, bS, c_tile); 
    
    auto c_tile_trans = make_coord(blockIdx.y, blockIdx.x);
    Tensor gD_tile = local_tile(gD, bS, c_tile_trans); 

    // 6. 核心修正点：Partition 逻辑
    // Global -> Smem (按行读，按行存入 Smem)
    Tensor tSgS = thr_copy.partition_S(gS_tile); 
    Tensor tSsS = thr_copy.partition_D(sS);      
    
    // Smem -> Global (关键：重新定义 Smem 的视图为转置视图，再进行 Partition)
    // 我们将 32x32 的 Smem 视为 (N=32, M=32) 的 ColMajor 布局，这样读取逻辑就转置了
    auto sS_trans = make_tensor(sS.data(), make_layout(bS, GenColMajor{}));
    // 这里依然使用 thr_copy，但作用在转置后的视图上
    Tensor tDsS = thr_copy.partition_S(sD); 
    Tensor tDgD = thr_copy.partition_D(gD_tile);

    // 7. 执行同步拷贝
    copy(tiled_copy, tSgS, tSsS);
    
    __syncthreads();

    // 此时 tDsS 已经包含了逻辑上的转置坐标映射
    copy(tiled_copy, tDsS, tDgD);
}

// ============================================================================
// CPU 参考实现
// ============================================================================
void transpose_cpu(const float* input, float* output, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      output[j * M + i] = input[i * N + j];
    }
  }
}

// ============================================================================
// 初始化数据
// ============================================================================
void init_data(float* data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = i;
    // static_cast<float>(rand()) / RAND_MAX;
  }
}

// ============================================================================
// 验证结果
// ============================================================================
bool verify_result(const float* a, const float* b, int size, float eps = 1e-5f) {
  for (int i = 0; i < size; ++i) {
    if (std::fabs(a[i] - b[i]) > eps) {
      printf("Mismatch at index %d: expected %.6f, got %.6f\n", i, a[i], b[i]);
      return false;
    }
  }
  return true;
}

// ============================================================================
// 计时工具
// ============================================================================
float get_elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}

// ============================================================================
// 主测试函数
// ============================================================================
int main(int argc, char** argv) {
  // 默认矩阵大小 M x N
  int M = 4096;
  int N = 4096;
  int warmup_iters = 10;
  int test_iters = 100;

  // 解析命令行参数
  if (argc >= 3) {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
  }
  if (argc >= 4) {
    test_iters = atoi(argv[3]);
  }

  printf("Matrix Transpose Test\n");
  printf("Matrix size: %d x %d\n", M, N);
  printf("Warmup iterations: %d\n", warmup_iters);
  printf("Test iterations: %d\n", test_iters);

  size_t input_size = M * N * sizeof(float);
  size_t output_size = N * M * sizeof(float);

  // 分配主机内存
  float* h_input = (float*)malloc(input_size);
  float* h_output = (float*)malloc(output_size);
  float* h_ref = (float*)malloc(output_size);

  // 初始化输入数据
  srand(42);
  init_data(h_input, M * N);

  // 分配设备内存
  float *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, input_size));
  CUDA_CHECK(cudaMalloc(&d_output, output_size));

  // 拷贝输入数据到设备
  CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));

  // 创建 CUDA 事件用于计时
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // ========== Warmup ==========
  printf("\nWarming up...\n");
  for (int i = 0; i < warmup_iters; ++i) {
    // TODO: 调用你的核函数
    // your_transpose_kernel<<<grid, block>>>(d_input, d_output, M, N);
    // CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // ========== Performance Test ==========
  printf("Running performance test...\n");
  CUDA_CHECK(cudaEventRecord(start));
const int tile_size = 32;
  // 每个 TiledCopy 我们用了 32 个线程 (8x4)
  dim3 block(32); 
  dim3 grid((M + tile_size - 1) / tile_size, (N + tile_size - 1) / tile_size);

  for (int i = 0; i < test_iters; ++i) {
    // TODO: 调用你的核函数
    cute_tiled_transpose_kernel<float, 4096, 4096><<<grid, block>>>(d_input, d_output);    // CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = get_elapsed_ms(start, stop);
  float avg_ms = total_ms / test_iters;

  // 计算带宽
  // 读 + 写 = 2 * M * N * sizeof(float) bytes
  double bytes_transferred = 2.0 * M * N * sizeof(float);
  double bandwidth_gbps = (bytes_transferred / (avg_ms * 1e-3)) / 1e9;

  printf("\n========== Results ==========\n");
  printf("Average time: %.4f ms\n", avg_ms);
  printf("Effective bandwidth: %.2f GB/s\n", bandwidth_gbps);

  // ========== Correctness Check ==========
  printf("\nVerifying correctness...\n");

  // 计算 CPU 参考结果
  transpose_cpu(h_input, h_ref, M, N);

  // 拷贝 GPU 结果回主机
  CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));

  // 验证
  if (verify_result(h_output, h_ref, M * N)) {
    printf("✓ Verification PASSED!\n");
  } else {
    printf("✗ Verification FAILED!\n");
  }

  // ========== Cleanup ==========
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  free(h_input);
  free(h_output);
  free(h_ref);

  printf("\nDone.\n");
  return 0;
}