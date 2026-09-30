import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.memory import SmemAllocator

@cute.struct
class SharedStorage:
    barriers: cute.struct.MemRange[cutlass.Int64, 2]
    slot: cute.struct.MemRange[cutlass.Int64, 1]

@cute.kernel
def mbarrier_kernel(output: cute.Tensor):
    storage = SmemAllocator().allocate(SharedStorage)
    full = storage.barriers.data_ptr()
    empty = full + 1
    slot = storage.slot.get_tensor(cute.make_layout(1))

    warp = cute.arch.warp_idx()
    lane = cute.arch.lane_idx()
    if warp == 0 and lane == 0:
        cute.arch.mbarrier_init(full, 1)
        cute.arch.mbarrier_init(empty, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    if warp == 0 and lane == 0:
        phase = cute.Int32(0)
        for i in range(4):
            if i > 0:
                cute.arch.mbarrier_wait(empty, phase ^ 1)
            slot[0] = i + 10
            cute.arch.mbarrier_arrive(full)
            phase = phase ^ 1

    if warp == 1 and lane == 0:
        phase = cute.Int32(0)
        for i in range(4):
            cute.arch.mbarrier_wait(full, phase ^ 1)
            output[i] = slot[0]
            cute.arch.mbarrier_arrive(empty)
            phase = phase ^ 1

def launch_mbarrier_kernel(output: cute.Tensor):
    mbarrier_kernel(output).launch(grid=(1,1,1), block=(64, 1, 1))

def main():
    import torch
    output = torch.empty(4, dtype=torch.int32, device="cuda")
    tensor = from_dlpack(output)
    compiled = cute.compile(launch_mbarrier_kernel, tensor)
    compiled(tensor)
    torch.cuda.synchronize()
    print(output)

