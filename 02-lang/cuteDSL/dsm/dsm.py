import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator

@cute.struct
class SharedStorage:
    tile: cute.struct.MemRange[cutlass.Int32, 1]

@cute.kernel
def dsm_kernel(output: cute.Tensor, cluster_size: cutlass.Constexpr):
    rank = cute.arch.block_idx_in_cluster()
    lane = cute.arch.thread_idx()[0]
    storage = SmemAllocator().allocate(SharedStorage)
    local_tile = storage.tile.data_ptr()

    if lane == 0:
        local_tile[0] = 100 + rank
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()

    if lane == 0:
        for peer in range(cluster_size):
            remote_tile = cute.arch.map_dsmem_ptr(local_tile, cutlass.Int32(peer))
            value = remote_tile[0]
            output[rank * cluster_size + peer] = value

    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()


