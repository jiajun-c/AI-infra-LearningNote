#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <vector>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cublas_v2.h>
#include <cute/tensor.hpp>

#include "cute/layout.hpp"
#include "cute/underscore.hpp"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device_2buffer(ProblemShape shape_MNK, CtaTiler cta_tiler,
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
  if (blockIdx.y == 0 && blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    print(sA);print("\n");print(tA); print("\n");
    print(tAsA);print("\n");
  }
  Tensor tBgB = local_partition(gB, tB, threadIdx.x);                  
  Tensor tBsB = local_partition(sB, tB, threadIdx.x);             

  auto tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   
  auto tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   
  if (blockIdx.y == 0 && blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    print(sA);print("\n");print(tC); print("\n");
    print(tCsA);print("\n");
  }
  // 输出
//   Using device 0: NVIDIA A800 80GB PCIe  (SM80, 108 SMs)
// smem_ptr[32b](0x7f8839000000) o (_128,_8,_2):(_1,_128,_1024)
// (_16,_16):(_1,_16)
  Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   
  Tensor tCrC = make_tensor_like(tCgC);                                
  clear(tCrC);
  auto K_TILE_MAX = size<2>(tAgA);
  int write_stage = 0;
  int read_stage  = 0;

  // --- Prologue: Load k=0 ---
  copy(tAgA(_,_,0), tAsA(_, _, write_stage));      
  copy(tBgB(_,_,0), tBsB(_, _, write_stage));      
  
  cp_async_fence();        
  write_stage ^= 1; 
  for (int k_tile = 0; k_tile < K_TILE_MAX-1; ++k_tile)
  {
      copy(tAgA(_,_,k_tile+1), tAsA(_, _, write_stage));      
      copy(tBgB(_,_,k_tile+1), tBsB(_, _, write_stage));      
      cp_async_fence();        

    // Wait Current
      cp_async_wait<1>();      
      __syncthreads();
      gemm(tCsA(_, _, read_stage), tCsB(_, _, read_stage), tCrC);

      __syncthreads();
    
      write_stage ^= 1;
      read_stage ^= 1;
  //   cp_async_fence();        
  //   cp_async_wait<0>();      
  //   __syncthreads();         

  //   gemm(tCsA, tCsB, tCrC);            
  //   __syncthreads();         
  }
// Epilogue
  cp_async_wait<0>();
  __syncthreads();
  gemm(tCsA(_,_,read_stage), tCsB(_,_,read_stage), tCrC);
  axpby(alpha, tCrC, beta, tCgC);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt_doubleBuffer(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
  using namespace cute;
  auto prob_shape = make_shape(m, n, k);
  auto cta_tiler = make_shape(Int<128>{}, Int<128>{}, Int<8>{});
  auto sA = make_layout(make_shape(Int<128>{}, Int<8>{}, Int<2>{}), make_stride(Int<1>{}, Int<128>{}, Int<1024>{})); 
  auto sB = make_layout(make_shape(Int<128>{}, Int<8>{}, Int<2>{}), make_stride(Int<1>{}, Int<128>{}, Int<1024>{})); 
  auto dA = make_stride(Int<1>{}, ldA);
  auto dB = make_stride(Int<1>{}, ldB);
  auto dC = make_stride(Int<1>{}, ldC);
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));
  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(m, 128)), size(ceil_div(n, 128)));
  gemm_device_2buffer<<<dimGrid, dimBlock, 0, stream>>>(prob_shape, cta_tiler, A, dA, sA, tA, B, dB, sB, tB, C, dC, make_layout(make_shape(Int<128>{}, Int<128>{})), tC, alpha, beta);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_double_buffer(char transA, char transB, int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
  if (transA == 'N' && transB == 'T') gemm_nt_doubleBuffer(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  // else if (transA == 'T' && transB == 'N') gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
}

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

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>(1);
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>(1);
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

  d_C = h_C; 
  gemm_double_buffer(transA, transB, m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta, d_C.data().get(), ldC);

  h_C = d_C;

  printf("%f\n", h_C[0]);
  CUTE_CHECK_LAST();
}