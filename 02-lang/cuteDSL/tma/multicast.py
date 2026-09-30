import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
CLUSTER_SIZE = 4
TILE = 128
NUM_TILES = 8

@cute.struct
class SharedStorage:
    barriers: cute.struct.MemRange[cutlass.Int64, 1]
    tiles:cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 128]]

@cute.kernel
def multicast_kernel(atom: cute.CopyAtom, g_src: cute.Tensor, out: cute.Tensor):
    rank = cute.arch.block_idx_in_cluster()
    tile_idx = cute.arch.block_idx()[0]//CLUSTER_SIZE
    lane = cute.arch.thread_idx()[0]
    storage = SmemAllocator().allocate(SharedStorage)
    barrier = storage.barriers.data_ptr()
    s_tile = storage.tile.get_tensor(cute.make_layout(TILE))
    g_tile = cute.local_tile(g_src, (TILE, ), (None, ))

    cta_layout = cute.make_layout(1)
    tma_s, tma_g = cute.nvgpu.cpasync.tma_partition(
        atom, 0, cta_layout, s_tile, g_tile
    )

    with cute.arch.elect_one():
        cute.arch.mbarrier_init(barrier, 1)
        cute.arch.mbarrier_expect_tx(barrier, TILE*4)
    cute.arch.mbarrier_init_fence()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()

    if rank == 0:
        cute.copy(
            atom.with_(
                tma_bar_ptr=barrier,
                mcast_mask=cutlass.Int16((1<<CLUSTER_SIZE)-1),
            ),tma_g,tma_s,)
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(barrier)
    cute.arch.mbarrier_wait(barrier, 0)

    for i in range(TILE/32):
        out[(tile_idx * CLUSTER_SIZE + rank) * TILE + i * 32 + lane] = s_tile[i*32+lane]

@cute.jit
def launch(src: cute.Tensor, out: cute.Tensor):
    atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(),
        src,
        cute.make_layout(TILE),
        (TILE, ),
    )
    multicast_kernel(atom, tma_tensor, out).launch(
        grid=(NUM_TILES * CLUSTER_SIZE, 1, 1),
        block=(32, 1, 1),
        cluster=(CLUSTER_SIZE, 1, 1),
    )