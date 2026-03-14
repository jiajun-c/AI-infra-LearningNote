#include "__clang_cuda_builtin_vars.h"
#include "__clang_cuda_runtime_wrapper.h"
#include "cute/algorithm/clear.hpp"
#include "cute/algorithm/copy.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/tuple_algorithms.hpp"
#include "cute/layout.hpp"
#include "cute/pointer.hpp"
#include "cute/pointer_flagged.hpp"
#include "cute/tensor_impl.hpp"
#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

template<class ProblemShape, class CtaTiler,
        class TA_ptr, class AStride, class ASmemLayout, class AThreadLayout,
        class TB_ptr, class BStride, class BSmemLayout, class BThreadLayout,
        class TC_ptr, class CStride, class CThreadLayout>
__global__ void 
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA_ptr A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB_ptr B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC_ptr C, CStride dC, CThreadLayout tC) {

    auto mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
    auto mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
    auto mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    auto gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    auto gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    auto gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    __shared__ float smemA[cosize_v<ASmemLayout>];
    __shared__ float smemB[cosize_v<BSmemLayout>];

    auto sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    auto sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    auto tAgA = local_partition(gA, tA, threadIdx.x);
    auto tAsA = local_partition(sA, tA, threadIdx.x);

    auto tBgB = local_partition(gB, tB, threadIdx.x);
    auto tBsB = local_partition(sB, tB, threadIdx.x);

    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});  
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});  
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   
    
    Tensor tCrC = make_tensor_like(tCgC);
    clear(tCrC);

    auto K_TILE_MAX = size<2>(tAgA);

    for (int k = 0; k < K_TILE_MAX; k++) {
        copy(tAgA(_, _, k), tAsA);
        copy(tBgB(_, _, k), tBsB);
        __syncthreads(); 
        gemm(tCsA, tCsB, tCrC);
        __syncthreads();
    }
    copy(tCrC, tCgC);
}