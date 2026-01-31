#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cublas_v2.h> // [新增] 引入 cuBLAS 头文件
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

// ============================================================================================
// ORIGINAL DEVICE KERNEL (Uses cp.async primitives)
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

  // ... (Preconditions and Static Asserts omitted for brevity, same as original) ...
  // Full and Tiled Tensors
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
    // Copy gmem to smem with tA|tB thread-partitioned tensors
    copy(tAgA(_,_,k_tile), tAsA);      
    copy(tBgB(_,_,k_tile), tBsB);      
    
    // ASYNC COPY PRIMITIVES
    cp_async_fence();        
    cp_async_wait<0>();      
    __syncthreads();         

    // Compute gemm on tC thread-partitioned smem
    gemm(tCsA, tCsB, tCrC);            

    __syncthreads();         
  }

  axpby(alpha, tCrC, beta, tCgC);
}

// ============================================================================================
// NEW: SYNC DEVICE KERNEL (NO cp.async primitives)
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

  // ... (Identical setup code) ...
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
    // Standard synchronous copy
    copy(tAgA(_,_,k_tile), tAsA);      
    copy(tBgB(_,_,k_tile), tBsB);      
    
    // *** MODIFICATION: REMOVED cp_async_fence() AND cp_async_wait() ***
    // We still need __syncthreads() to ensure data is in Shared Memory before GEMM
    __syncthreads();         

    // Compute gemm
    gemm(tCsA, tCsB, tCrC);            

    __syncthreads();         
  }

  axpby(alpha, tCrC, beta, tCgC);
}

// ============================================================================================
// WRAPPERS (Async & Sync)
// ============================================================================================

// --- Original NT GEMM (Async) ---
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0)
{
  using namespace cute;
  auto M = int(m); auto N = int(n); auto K = int(k);
  auto prob_shape = make_shape(M, N, K);
  auto dA = make_stride(Int<1>{}, ldA);
  auto dB = make_stride(Int<1>{}, ldB);
  auto dC = make_stride(Int<1>{}, ldC);
  auto bM = Int<128>{}; auto bN = Int<128>{}; auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto sA = make_layout(make_shape(bM, bK));
  auto sB = make_layout(make_shape(bN, bK));
  auto sC = make_layout(make_shape(bM, bN));
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));
  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA, sA, tA, B, dB, sB, tB, C, dC, sC, tC, alpha, beta);
}

// --- NEW NT GEMM (Sync) ---
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt_sync(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0)
{
  using namespace cute;
  auto M = int(m); auto N = int(n); auto K = int(k);
  auto prob_shape = make_shape(M, N, K);
  auto dA = make_stride(Int<1>{}, ldA);
  auto dB = make_stride(Int<1>{}, ldB);
  auto dC = make_stride(Int<1>{}, ldC);
  auto bM = Int<128>{}; auto bN = Int<128>{}; auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto sA = make_layout(make_shape(bM, bK));
  auto sB = make_layout(make_shape(bN, bK));
  auto sC = make_layout(make_shape(bM, bN));
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));
  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  // Calls gemm_device_sync
  gemm_device_sync<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA, sA, tA, B, dB, sB, tB, C, dC, sC, tC, alpha, beta);
}

// --- Original TN GEMM (Async) ---
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_tn(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0)
{
  using namespace cute;
  auto M = int(m); auto N = int(n); auto K = int(k);
  auto prob_shape = make_shape(M, N, K);
  auto dA = make_stride(ldA, Int<1>{});
  auto dB = make_stride(ldB, Int<1>{});
  auto dC = make_stride(Int<1>{}, ldC);
  auto bM = Int<128>{}; auto bN = Int<128>{}; auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto sA = make_layout(make_shape(bM,bK), LayoutRight{});
  auto sB = make_layout(make_shape(bN,bK), LayoutRight{});
  auto sC = make_layout(make_shape(bM, bN));
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));
  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA, sA, tA, B, dB, sB, tB, C, dC, sC, tC, alpha, beta);
}

// --- NEW TN GEMM (Sync) ---
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_tn_sync(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0)
{
  using namespace cute;
  auto M = int(m); auto N = int(n); auto K = int(k);
  auto prob_shape = make_shape(M, N, K);
  auto dA = make_stride(ldA, Int<1>{});
  auto dB = make_stride(ldB, Int<1>{});
  auto dC = make_stride(Int<1>{}, ldC);
  auto bM = Int<128>{}; auto bN = Int<128>{}; auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto sA = make_layout(make_shape(bM,bK), LayoutRight{});
  auto sB = make_layout(make_shape(bN,bK), LayoutRight{});
  auto sC = make_layout(make_shape(bM, bN));
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));
  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  // Calls gemm_device_sync
  gemm_device_sync<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA, sA, tA, B, dB, sB, tB, C, dC, sC, tC, alpha, beta);
}

// Wrapper for Async
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm(char transA, char transB, int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0)
{
  if (transA == 'N' && transB == 'T') {
    return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  } else if (transA == 'T' && transB == 'N') {
    return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  }
  assert(false && "Not implemented");
}

// Wrapper for Sync
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_sync(char transA, char transB, int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0)
{
  if (transA == 'N' && transB == 'T') {
    return gemm_nt_sync(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  } else if (transA == 'T' && transB == 'N') {
    return gemm_tn_sync(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  }
  assert(false && "Not implemented");
}


int main(int argc, char** argv)
{
  int m = 5120;
  if (argc >= 2) sscanf(argv[1], "%d", &m);
  int n = 5120;
  if (argc >= 3) sscanf(argv[2], "%d", &n);
  int k = 4096;
  if (argc >= 4) sscanf(argv[3], "%d", &k);
  char transA = 'N';
  if (argc >= 5) sscanf(argv[4], "%c", &transA);
  char transB = 'T';
  if (argc >= 6) sscanf(argv[5], "%c", &transB);

  using TA = float;
  using TB = float;
  using TC = float;
  using TI = float;

  TI alpha = 1.0;
  TI beta  = 0.0;

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
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

  // ----------------------------------------------------------------
  // 1. Run CUTE_GEMM (Original - Async)
  // ----------------------------------------------------------------

  // ----------------------------------------------------------------
  // 2. Run CUTE_GEMM_SYNC (New - No Async)
  // ----------------------------------------------------------------
  d_C = h_C; // Reset C
  gemm_sync(transA, transB, m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
  CUTE_CHECK_LAST();

  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm_sync(transA, transB, m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
  }
  double sync_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM (Sync) : [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / sync_time, sync_time*1000);

  d_C = h_C;
  gemm(transA, transB, m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
  CUTE_CHECK_LAST();
  
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm(transA, transB, m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(  d_C = h_C;
  gemm(transA, transB, m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
  CUTE_CHECK_LAST();
  
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm(transA, transB, m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM (Async): [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);), ldB, beta, d_C.data().get(), ldC);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM (Async): [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);
  return 0;
}