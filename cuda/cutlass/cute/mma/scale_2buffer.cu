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
// 1. DEVICE KERNEL: Single Buffer (Synchronous)
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

  // Tensors
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

  // [Vectorization Logic]
  // Detect stride 1 dimension to form float4 (128-bit) vector loads.
  using A_Vec_Shape = conditional_t<is_same_v<decltype(get<0>(dA)), Int<1>>, Shape<_4, _1>, Shape<_1, _4>>;
  using B_Vec_Shape = conditional_t<is_same_v<decltype(get<0>(dB)), Int<1>>, Shape<_4, _1>, Shape<_1, _4>>;

  // [Change 1]: Use UniversalCopy instead of SM80_CP_ASYNC
  // This performs a synchronous copy: Global -> Register -> Shared
  // We explicitly use uint128_t to enforce 128-bit vectorized load/store instructions (LDG.128 / STS.128)
  using CopyOp = UniversalCopy<cute::uint128_t>; 

  auto tAgA_copy_strategy = make_tiled_copy(
      Copy_Atom<CopyOp, TA>{},
      tA,
      Layout<A_Vec_Shape>{} 
  );
  auto tBgB_copy_strategy = make_tiled_copy(
      Copy_Atom<CopyOp, TB>{},
      tB,
      Layout<B_Vec_Shape>{}
  );

  Tensor tAgA_copy_src = tAgA_copy_strategy.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
  Tensor tAsA_copy_dst = tAgA_copy_strategy.partition_D(sA); // (CPY, CPY_M, CPY_K)
  Tensor tBgB_copy_src = tBgB_copy_strategy.partition_S(gB); 
  Tensor tBsB_copy_dst = tBgB_copy_strategy.partition_D(sB); 

  // Compute Partitions
  Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   
  Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   
  Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   
  Tensor tCrC = make_tensor_like(tCgC);                                
  clear(tCrC);

  auto K_TILE_MAX = size<2>(tAgA_copy_src);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    // [Change 2]: Synchronous Copy
    // This function will BLOCK until the copy is finished by this thread.
    cute::copy(tAgA_copy_strategy, tAgA_copy_src(_,_,_,k_tile), tAsA_copy_dst);
    cute::copy(tBgB_copy_strategy, tBgB_copy_src(_,_,_,k_tile), tBsB_copy_dst);
    
    // [Change 3]: Removed cp_async_fence / cp_async_wait
    
    // Barrier is still needed to ensure all threads finished writing to Smem
    // before any thread starts reading from Smem.
    __syncthreads();         

    gemm(tCsA, tCsB, tCrC); 
    
    // Barrier needed to ensure all threads finished reading from Smem
    // before we overwrite it in the next iteration.
    __syncthreads();         
  }

  axpby(alpha, tCrC, beta, tCgC);
}

// ============================================================================================
// 2. DEVICE KERNEL: Double Buffer (Synchronous)
// ============================================================================================
// Even without cp.async, double buffering can help if we had independent instruction streams,
// but in a single warp, this mostly serves to demonstrate the buffer rotation logic.
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device_double_buffer(ProblemShape shape_MNK, CtaTiler cta_tiler,
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

  // Shared Memory: (BLK_M, BLK_K, Stages=2)
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            

  using A_Vec_Shape = conditional_t<is_same_v<decltype(get<0>(dA)), Int<1>>, Shape<_4, _1>, Shape<_1, _4>>;
  using B_Vec_Shape = conditional_t<is_same_v<decltype(get<0>(dB)), Int<1>>, Shape<_4, _1>, Shape<_1, _4>>;
  
  // Use UniversalCopy (Synchronous)
  using CopyOp = UniversalCopy<cute::uint128_t>; 

  auto tAgA_copy_strategy = make_tiled_copy(
      Copy_Atom<CopyOp, TA>{},
      tA,
      Layout<A_Vec_Shape>{}
  );
  auto tBgB_copy_strategy = make_tiled_copy(
      Copy_Atom<CopyOp, TB>{},
      tB,
      Layout<B_Vec_Shape>{}
  );

  Tensor tAgA_copy_src = tAgA_copy_strategy.partition_S(gA); // (CPY, M, K, k)
  Tensor tBgB_copy_src = tBgB_copy_strategy.partition_S(gB); 
  
  Tensor tAsA_copy_dst = tAgA_copy_strategy.partition_D(sA); // (CPY, M, K, Stages)
  Tensor tBsB_copy_dst = tBgB_copy_strategy.partition_D(sB); 

  Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   
  Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   
  Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   
  Tensor tCrC = make_tensor_like(tCgC);                                
  clear(tCrC);

  auto K_TILE_MAX = size<2>(tAgA_copy_src);

  // --- Double Buffer Prologue ---
  // Load the first tile (k=0) into stage 0
  int write_stage = 0;
  cute::copy(tAgA_copy_strategy, tAgA_copy_src(_,_,_,0), tAsA_copy_dst(_,_,_,write_stage));
  cute::copy(tBgB_copy_strategy, tBgB_copy_src(_,_,_,0), tBsB_copy_dst(_,_,_,write_stage));
  
  // Wait for load to finish
  __syncthreads(); 

  int read_stage = 0;
  write_stage = 1; 

  // --- Main Loop ---
  for (int k_tile = 0; k_tile < K_TILE_MAX - 1; ++k_tile)
  {
    // 1. Compute on current tile (read_stage)
    // Note: In sync mode, we usually compute first, then load next, 
    // or load next then compute. Since load is blocking, "overlap" is limited.
    gemm(tCsA(_,_,read_stage), tCsB(_,_,read_stage), tCrC);

    // 2. Ensure we are done reading from read_stage before it becomes the write_stage later
    // (Though in this specific loop structure, read_stage != write_stage, so simple Double Buffer is safe)
    
    // 3. Load NEXT tile (k+1) into write_stage
    // This call BLOCKS until data is in Shared Memory
    cute::copy(tAgA_copy_strategy, tAgA_copy_src(_,_,_,k_tile+1), tAsA_copy_dst(_,_,_,write_stage));
    
    // 4. Barrier to make sure the new data is visible for the next iteration's compute
    __syncthreads(); 

    // 5. Flip buffers
    write_stage ^= 1;
    read_stage  ^= 1;
  }

  // --- Epilogue ---
  // Compute the last tile
  gemm(tCsA(_,_,read_stage), tCsB(_,_,read_stage), tCrC);

  axpby(alpha, tCrC, beta, tCgC);
}

// ============================================================================================
// WRAPPERS
// ============================================================================================

// --- Original NT GEMM ---
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
  using namespace cute;
  auto prob_shape = make_shape(m, n, k);
  auto cta_tiler = make_shape(Int<128>{}, Int<128>{}, Int<8>{});
  auto sA = make_layout(make_shape(Int<128>{}, Int<8>{})); // Col-Major blocks
  auto sB = make_layout(make_shape(Int<128>{}, Int<8>{})); // Col-Major blocks
  
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

// --- DOUBLE BUFFER NT GEMM ---
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt_double_buffer(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
  using namespace cute;
  auto prob_shape = make_shape(m, n, k);
  auto cta_tiler = make_shape(Int<128>{}, Int<128>{}, Int<8>{});
  
  auto bM = Int<128>{}; auto bN = Int<128>{}; auto bK = Int<8>{};
  auto sA = make_layout(make_shape(bM, bK, Int<2>{}), make_stride(Int<1>{}, bM, bM*bK)); 
  auto sB = make_layout(make_shape(bN, bK, Int<2>{}), make_stride(Int<1>{}, bN, bN*bK)); 

  auto dA = make_stride(Int<1>{}, ldA);
  auto dB = make_stride(Int<1>{}, ldB);
  auto dC = make_stride(Int<1>{}, ldC);
  
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(m, 128)), size(ceil_div(n, 128)));
  gemm_device_double_buffer<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA, sA, tA, B, dB, sB, tB, C, dC, make_layout(make_shape(Int<128>{}, Int<128>{})), tC, alpha, beta);
}

// --- Original TN GEMM ---
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

// --- DOUBLE BUFFER TN GEMM ---
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_tn_double_buffer(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
  using namespace cute;
  auto prob_shape = make_shape(m, n, k);
  auto cta_tiler = make_shape(Int<128>{}, Int<128>{}, Int<8>{});
  
  auto bM = Int<128>{}; auto bN = Int<128>{}; auto bK = Int<8>{};
  auto sA = make_layout(make_shape(bM, bK, Int<2>{}), make_stride(bK, Int<1>{}, bM*bK)); 
  auto sB = make_layout(make_shape(bN, bK, Int<2>{}), make_stride(bK, Int<1>{}, bN*bK)); 

  auto dA = make_stride(ldA, Int<1>{});
  auto dB = make_stride(ldB, Int<1>{});
  auto dC = make_stride(Int<1>{}, ldC);
  
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(m, 128)), size(ceil_div(n, 128)));
  gemm_device_double_buffer<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA, sA, tA, B, dB, sB, tB, C, dC, make_layout(make_shape(Int<128>{}, Int<128>{})), tC, alpha, beta);
}

// Dispatchers
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm(char transA, char transB, int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
  if (transA == 'N' && transB == 'T') gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  else if (transA == 'T' && transB == 'N') gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_double_buffer(char transA, char transB, int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
  if (transA == 'N' && transB == 'T') gemm_nt_double_buffer(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  else if (transA == 'T' && transB == 'N') gemm_tn_double_buffer(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
}

// ============================================================================================
// MAIN
// ============================================================================================
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

  using TA = float; using TB = float; using TC = float; using TI = float;
  TI alpha = 1.0; TI beta  = 0.0;

  std::cout << "M = " << m << ", N = " << n << ", K = " << k << std::endl;
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

  // 1. Single Buffer
  d_C = h_C; 
  gemm(transA, transB, m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
  CUTE_CHECK_LAST();
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm(transA, transB, m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
  }
  double time_sb = timer.seconds() / timing_iterations;
  printf("GEMM (1-Buff)    : [%6.1f] GFlop/s  (%6.4f)ms\n", gflops / time_sb, time_sb*1000);

  // 2. Double Buffer 
  d_C = h_C; 
  gemm_double_buffer(transA, transB, m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
  CUTE_CHECK_LAST();
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm_double_buffer(transA, transB, m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
  }
  double time_db = timer.seconds() / timing_iterations;
  printf("GEMM (2-Buff)    : [%6.1f] GFlop/s  (%6.4f)ms\n", gflops / time_db, time_db*1000);

  // 3. cuBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasOperation_t opA = (transA == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t opB = (transB == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;

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