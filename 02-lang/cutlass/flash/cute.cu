#include <cuda.h>
#include <cute/tensor.hpp>

using namespace cute;

template<typename T, int BlockQO, int BlockKV, int HeadDim, int NumThreads>
__global__ void flash_attn_kernel(T* pQ, T* pK, T* pV, T* pO, int B, int H, int N, float scaler) {
    auto Q = make_tensor(make_gmem_ptr(pQ), make_layout(make_shape(B, H, N, Int<HeadDim>{}), GenRowMajor{}));
    auto K = make_tensor(make_gmem_ptr(pK), make_layout(make_shape(B, H, N, Int<HeadDim>{}), GenRowMajor{}));
    auto V = make_tensor(make_gmem_ptr(pV), make_layout(make_shape(B, H, N, Int<HeadDim>{}), GenRowMajor{}));
    auto O = make_tensor(make_gmem_ptr(pO), make_layout(make_shape(B, H, N, Int<HeadDim>{}), GenRowMajor{}));

    const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    const int tx = threadIdx.x;

    // 分块的shape为(1, 1, BLOCK_QO, HeadDim) 
    // 切出来的分块数量为 (B, H, BLOCK_QO, HeadDim, BlockNum)
    // 选择的分块shape为 (bx, by, bz)
    // gQ [1, 1, BlockQO, HeadDim]
    // gK [1, 1, BlockKV, HeadDim, TILE_NUM] -> ()
    auto gQ = local_tile(Q, make_shape(_1{}, _1{}, Int<BlockQO>{}, Int<HeadDim>{}), make_coord(bx, by, bz, 0))(0, 0, _, _);
    auto gO = local_tile(O, make_shape(_1{}, _1{}, Int<BlockQO>{}, Int<HeadDim>{}), make_coord(bx, by, bz, 0))(0, 0, _, _);
    auto gK = local_tile(K, make_shape(_1{}, _1{}, Int<BlockKV>{}, Int<HeadDim>{}), make_coord(bx, by, _, 0))(0, 0, _, _, _);
    auto gV = local_tile(V, make_shape(_1{}, _1{}, Int<BlockKV>{}, Int<HeadDim>{}), make_coord(bx, by, _, 0))(0, 0, _, _, _);

    __shared__ T psQ[BlockQO * HeadDim];
    __shared__ T psK[BlockKV * HeadDim];
    __shared__ T psV[BlockKV * HeadDim];

    auto sQ = make_tensor(make_smem_ptr(psQ), make_layout(make_shape(Int<BlockQO>{}, Int<HeadDim>{}), GenRowMajor{}));
    auto sK = make_tensor(make_smem_ptr(psK), make_layout(make_shape(Int<BlockKV>{}, Int<HeadDim>{}), GenRowMajor{}));
    auto sV = make_tensor(make_smem_ptr(psV), make_layout(make_shape(Int<BlockKV>{}, Int<HeadDim>{}), GenRowMajor{}));
    auto sVt = make_tensor(make_smem_ptr(psV), make_layout(make_shape(Int<HeadDim>{}, Int<BlockKV>{}), GenColMajor{}));

    using GmemCopyAtom = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, T>;
    auto tiled_copy = make_tiled_copy(GmemCopyAtom{}, make_layout(Shape<Int<NumThreads/8, _8>{}, GenRowMajor{}>), Layout<Shape<_1, _8>>{});
    auto thr_copy = tiled_copy.get_slice(tx);


}
