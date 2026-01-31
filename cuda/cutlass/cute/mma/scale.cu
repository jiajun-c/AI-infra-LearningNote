#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <vector>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cublas_v2.h>
#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

// ============================================================================================
// 1. DEVICE KERNEL: Async (Original, uses cp.async)
// ============================================================================================
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * C, CStride dC, CSmemLayout          , CThreadLayout tC,
            Alpha alpha, Beta beta)
{
  using namespace cute;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); 
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); 
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); 

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  

  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            

  Tensor tAgA = local_partition(gA, tA, threadIdx.x);                  
  Tensor tAsA = local_partition(sA, tA, threadIdx.x);                  

  Tensor tBgB = local_partition(gB, tB, threadIdx.x);                  
  Tensor tBsB = local_partition(sB, tB, threadIdx.x);                  

  Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   
  Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   
  Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   
  Tensor tCrC = make_tensor_like(tCgC);                                
  clear(tCrC);

  auto K_TILE_MAX = size<2>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    copy(tAgA(_,_,k_tile), tAsA);      
    copy(tBgB(_,_,k_tile), tBsB);      
    
    cp_async_fence();        
    cp_async_wait<0>();      
    __syncthreads();         

    gemm(tCsA, tCsB, tCrC);            
    __syncthreads();         
  }

  axpby(alpha, tCrC, beta, tCgC);
}

// ============================================================================================
// 2. DEVICE KERNEL: Sync (No cp.async)
// ============================================================================================
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device_sync(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * C, CStride dC, CSmemLayout          , CThreadLayout tC,
            Alpha alpha, Beta beta)
{
  using namespace cute;

  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); 
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); 
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); 

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  

  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            

  Tensor tAgA = local_partition(gA, tA, threadIdx.x);                  
  Tensor tAsA = local_partition(sA, tA, threadIdx.x);                  

  Tensor tBgB = local_partition(gB, tB, threadIdx.x);                  
  Tensor tBsB = local_partition(sB, tB, threadIdx.x);                  

  Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   
  Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   
  Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   
  Tensor tCrC = make_tensor_like(tCgC);                                
  clear(tCrC);

  auto K_TILE_MAX = size<2>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    copy(tAgA(_,_,k_tile), tAsA);      
    copy(tBgB(_,_,k_tile), tBsB);      
    
    // REMOVED cp_async primitives
    __syncthreads();         

    gemm(tCsA, tCsB, tCrC);            
    __syncthreads();         
  }

  axpby(alpha, tCrC, beta, tCgC);
}

// ============================================================================================
// WRAPPERS
// ============================================================================================

// --- Async Wrappers ---
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
  using namespace cute;
  auto prob_shape = make_shape(m, n, k);
  auto cta_tiler = make_shape(Int<128>{}, Int<128>{}, Int<8>{});
  auto sA = make_layout(make_shape(Int<128>{}, Int<8>{})); 
  auto sB = make_layout(make_shape(Int<128>{}, Int<8>{})); 
  auto dA = make_stride(Int<1>{}, ldA);
  auto dB = make_stride(Int<1>{}, ldB);
  auto dC = make_stride(Int<1>{}, ldC);
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));
  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(m, 128)), size(ceil_div(n, 128)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA, sA, tA, B, dB, sB, tB, C, dC, make_layout(make_shape(Int<128>{}, Int<128>{})), tC, alpha, beta);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_tn(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
  using namespace cute;
  auto prob_shape = make_shape(m, n, k);
  auto cta_tiler = make_shape(Int<128>{}, Int<128>{}, Int<8>{});
  auto sA = make_layout(make_shape(Int<128>{}, Int<8>{}), LayoutRight{});
  auto sB = make_layout(make_shape(Int<128>{}, Int<8>{}), LayoutRight{});
  auto dA = make_stride(ldA, Int<1>{});
  auto dB = make_stride(ldB, Int<1>{});
  auto dC = make_stride(Int<1>{}, ldC);
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));
  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(m, 128)), size(ceil_div(n, 128)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA, sA, tA, B, dB, sB, tB, C, dC, make_layout(make_shape(Int<128>{}, Int<128>{})), tC, alpha, beta);
}

// --- Sync Wrappers ---
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt_sync(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
  using namespace cute;
  auto prob_shape = make_shape(m, n, k);
  auto cta_tiler = make_shape(Int<128>{}, Int<128>{}, Int<8>{});
  auto sA = make_layout(make_shape(Int<128>{}, Int<8>{})); 
  auto sB = make_layout(make_shape(Int<128>{}, Int<8>{})); 
  auto dA = make_stride(Int<1>{}, ldA);
  auto dB = make_stride(Int<1>{}, ldB);
  auto dC = make_stride(Int<1>{}, ldC);
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));
  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(m, 128)), size(ceil_div(n, 128)));
  gemm_device_sync<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA, sA, tA, B, dB, sB, tB, C, dC, make_layout(make_shape(Int<128>{}, Int<128>{})), tC, alpha, beta);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_tn_sync(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
  using namespace cute;
  auto prob_shape = make_shape(m, n, k);
  auto cta_tiler = make_shape(Int<128>{}, Int<128>{}, Int<8>{});
  auto sA = make_layout(make_shape(Int<128>{}, Int<8>{}), LayoutRight{});
  auto sB = make_layout(make_shape(Int<128>{}, Int<8>{}), LayoutRight{});
  auto dA = make_stride(ldA, Int<1>{});
  auto dB = make_stride(ldB, Int<1>{});
  auto dC = make_stride(Int<1>{}, ldC);
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));
  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(m, 128)), size(ceil_div(n, 128)));
  gemm_device_sync<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA, sA, tA, B, dB, sB, tB, C, dC, make_layout(make_shape(Int<128>{}, Int<128>{})), tC, alpha, beta);
}
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_double_buffer(char transA, char transB, int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
  if (transA == 'N' && transB == 'T') gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  else if (transA == 'T' && transB == 'N') gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
}
// Dispatchers
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm(char transA, char transB, int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
  if (transA == 'N' && transB == 'T') gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  else if (transA == 'T' && transB == 'N') gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_sync(char transA, char transB, int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
  if (transA == 'N' && transB == 'T') gemm_nt_sync(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  else if (transA == 'T' && transB == 'N') gemm_tn_sync(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
}


// ============================================================================================
// MAIN (Fixed Parameters)
// ============================================================================================
int main(int argc, char** argv)
{
  // --------------------------------------------------------
  // Fixed Configuration (No CLI args)
  // --------------------------------------------------------
  int m = 5120;
  int n = 5120;
  int k = 4096;
  char transA = 'N';
  char transB = 'T';
  // --------------------------------------------------------

  using TA = float; using TB = float; using TC = float; using TI = float;
  TI alpha = 1.0; TI beta  = 0.0;

  std::cout << "M = " << m << ", N = " << n << ", K = " << k << std::endl;
  std::cout << "C = A^" << transA << " B^" << transB << std::endl;
  
  cute::device_init(0);

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  thrust::host_vector<TC> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  double gflops = (2.0*m*n*k) * 1e-9;
  const int timing_iterations = 100;
  GPU_Clock timer;

  int ldA = 0, ldB = 0, ldC = m;
  if (transA == 'N') ldA = m; else if (transA == 'T') ldA = k; else assert(false);
  if (transB == 'N') ldB = k; else if (transB == 'T') ldB = n; else assert(false);

  // 1. Run CUTE_GEMM_SYNC (Synchronous)
  d_C = h_C; 
  gemm_sync(transA, transB, m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
  CUTE_CHECK_LAST();

  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm_sync(transA, transB, m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
  }
  double time_sync = timer.seconds() / timing_iterations;
  printf("GEMM (Sync)      : [%6.1f] GFlop/s  (%6.4f)ms\n", gflops / time_sync, time_sync*1000);

  // 2. Run CUTE_GEMM_ASYNC (Async cp.async)
  d_C = h_C; 
  gemm(transA, transB, m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
  CUTE_CHECK_LAST();

  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm(transA, transB, m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
  }
  double time_async = timer.seconds() / timing_iterations;
  printf("GEMM (Async)     : [%6.1f] GFlop/s  (%6.4f)ms\n", gflops / time_async, time_async*1000);

  // 3. Run cuBLAS (Baseline)
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasOperation_t opA = (transA == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t opB = (transB == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
  
  // Warmup
  cublasSgemm(handle, opA, opB, m, n, k, &alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, &beta, d_C.data().get(), ldC);

  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
      cublasSgemm(handle, opA, opB, m, n, k, &alpha,
                  d_A.data().get(), ldA,
                  d_B.data().get(), ldB,
                  &beta,
                  d_C.data().get(), ldC);
  }
  double time_cublas = timer.seconds() / timing_iterations;
  printf("cuBLAS           : [%6.1f] GFlop/s  (%6.4f)ms\n", gflops / time_cublas, time_cublas*1000);
  
  cublasDestroy(handle);

  return 0;
}