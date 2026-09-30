import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.memory import SmemAllocator
from cutlass.cute.runtime import from_dlpack

@cute.struct
class Storage:
    barriers: cute.struct.MemRange[cute.Int64, 4]
    data: cute.struct.MemRange[cute.Int32, 2]

@cute.kernel
def producer_consumer_demo(output: cute.Tensor, slot_indices: cute.Tensor):
    storage = SmemAllocator().allocate(Storage)
    barrier_ptr = storage.barriers.data_ptr()
    data = storage.data.get_tensor(cute.make_layout(2))
    producer, consumer = pipeline.PipelineAsync.create(
        barrier_storage=barrier_ptr,
        num_stages=2,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread)
    ).make_participants()

    warp = cute.arch.warp_idx()
    lane = cute.arch.lane_idx()

    if warp == 0 and lane == 0:
        for i in range(4):
            handle = producer.acquire_and_advance()
            slot = handle.index
            data[slot] = i + 10
            handle.commit()
        producer.tail()

    if warp == 1 and lane == 0:
        for i in range(4):
            handle = consumer.release_and_advance()
            slot = handle.index
            output[i] = data[slot]
            slot_indices[i] = slot
            handle.release()

@cute.jit
def launch_kernel(output: cute.Tensor, slot_indices: cute.Tensor):
    producer_consumer_demo(output, slot_indices).launch(
        grid=(1, 1, 1),
        block=(64, 1, 1)
    )

def main():
    import torch
    output = torch.empty(4, dtype=torch.int32, device="cuda")
    slot_indices = torch.empty(4, dtype=torch.int32, device="cuda")

    output_cute = from_dlpack(output)
    slot_indices_cute = from_dlpack(slot_indices)

    compiled = cute.compile(launch_kernel, output_cute, slot_indices_cute)

    compiled(output_cute, slot_indices_cute)
    torch.cuda.synchronize()

    print("output:", output.tolist())
    print("slot_indices:",slot_indices.tolist())
