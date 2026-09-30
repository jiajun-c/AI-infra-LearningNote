import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.memory import SmemAllocator

M, N = 256, 128
TILE_M, TILE_N = 64, 64
TILE_BYTES = TILE_M * TILE_N * 2

@cute.struct
class SharedStorage:
    barriers: cute.struct.MemRange[cutlass.Int64, 1]
    tile: cute.struct.Align[
        cute.struct.MemRange[cute.Float16, TILE_M * TILE_N], 128
    ]

@cute.kernel
def copy_2d_kernel(
    load_atom: cute.CopyAtom,
    load_tensor: cute.Tensor,
    store_atom: cute.CopyAtom,
    store_tensor: cute.Tensor
):
    block_m, block_n, _ = cute.arch.block_idx()
    storage = SmemAllocator().allocate(SharedStorage)
    barrier = storage.barrier.data_ptr()
    smem_layout = cute.make_layout((TILE_M, TILE_N), stride=(TILE_N, 1))
    s_tile = storage.tile.get_tensor(smem_layout)

    g_src_tiles = cute.local_tile(
        load_tensor, 
        (TILE_M, TILE_N),
        (None, None)
    )

    g_dst_tiles = cute.local_tile(
        store_tensor,
        (TILE_M, TILE_N),
        (None, None)
    )

    tma_s, tma_src = cute.nvgpu.cpasync.tma_partition(
        load_atom,
        0,
        cute.make_layout(1),
        cute.group_modes(s_tile, 0, 2),
        cute.group_modes(g_src_tiles, 0, 2)
    )

    _, tma_dst = cute.nvgpu.cpasync.tma_partition(
        store_atom,
        0,
        cute.make_layout(1),
        cute.group_modes(s_tile, 0, 2),
        cute.group_modes(g_dst_tiles, 0, 2)
    )

    with cute.arch.elect_one():
        cute.arch.mbarrier_init(barrier, 1)
        cute.arch.mbarrier_expect_tx(barrier, TILE_BYTES)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_warp()

    cute.copy(
        load_atom,
        tma_src[(None, block_m, block_n)],
        tma_s,
        tma_bar_ptr=barrier
    )

    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(barrier)
    cute.arch.mbarrier_wait(barrier, 0)

    cute.copy(store_atom, tma_s, tma_dst[(None, block_m, block_n)])

    cute.arch.cp_async_bulk_commit_group()
    cute.arch.cp_async_bulk_wait_group(0, read=False)

@cute.jit
def launch(src: cute.Tensor, dst: cute.Tensor):
    smem_layout = cute.make_layout((TILE_M, TILE_N), stride=(TILE_N, 1))
    load_atom, load_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        src,
        smem_layout,
        (TILE_M, TILE_N)
    )
    store_atom, store_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        dst,
        smem_layout,
        (TILE_M, TILE_N)
    )
    copy_2d_kernel(load_atom, load_tensor, store_atom, store_tensor).launch(
        grid = (M//TILE_M, N//TILE_N, 1),
        block = (32, 1, 1)
    )

def main():
    import torch
    