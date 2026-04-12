import os
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

def get_cuda_memory():
    # 返回当前分配的显存字节数
    return torch.cuda.memory_allocated()

def test_reduce_memory(size_mb, mode="standard"):
    num_elements = (size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
    rank = dist.get_rank()
    
    # 强制清理缓存，保证统计准确
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    base_mem = get_cuda_memory()

    if mode == "standard":
        # 1. 申请标准 Tensor (User Buffer)
        t = torch.zeros((num_elements,), device=f"cuda:{rank}")
        mem_after_alloc = get_cuda_memory()
        user_buffer_size = mem_after_alloc - base_mem
        
        # 2. 执行标准 All-Reduce
        dist.all_reduce(t)
        torch.cuda.synchronize()
        mem_after_op = get_cuda_memory()
        
    else:
        # 1. 申请对称内存 (User Buffer)
        t = symm_mem.empty((num_elements,), dtype=torch.float32, device=torch.device(f"cuda:{rank}"))
        hdl = symm_mem.rendezvous(t, dist.group.WORLD)
        mem_after_alloc = get_cuda_memory()
        user_buffer_size = mem_after_alloc - base_mem
        
        # 2. 执行 One-Shot All-Reduce
        group_name = dist._get_process_group_name(dist.group.WORLD)
        res = torch.ops.symm_mem.one_shot_all_reduce(t, "sum", group_name)
        torch.cuda.synchronize()
        mem_after_op = get_cuda_memory()

    return user_buffer_size, mem_after_op - mem_after_alloc

def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    # 测试不同数据量: 10MB, 100MB, 500MB, 1GB
    test_sizes = [10, 100, 500, 1024]
    
    if local_rank == 0:
        print(f"{'Data Size':>10} | {'Mode':>10} | {'User Buffer':>15} | {'Extra Comm Buffer':>20}")
        print("-" * 65)

    for size in test_sizes:
        for mode in ["standard", "symm_mem"]:
            user_size, extra_size = test_reduce_memory(size, mode)
            dist.barrier()
            if local_rank == 0:
                print(f"{size:>8} MB | {mode:>10} | {user_size/1024**2:>12.2f} MB | {extra_size/1024**2:>17.2f} MB")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()