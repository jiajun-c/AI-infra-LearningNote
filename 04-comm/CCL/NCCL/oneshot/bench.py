import os
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

def benchmark_op(func, iterations=100, warmup=20):
    # 预热
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    # 计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        func()
    end_event.record()
    
    torch.cuda.synchronize()
    # 返回平均耗时 (单位: 毫秒 -> 转换为 微秒)
    return (start_event.elapsed_time(end_event) / iterations) * 1000

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 测试规模：从 4KB (Decode 典型值) 到 16MB
    # 4KB, 64KB, 1MB, 4MB, 16MB
    sizes_in_bytes = [4 * 1024, 8*1024, 16*1024, 32*1024, 64 * 1024,128*1024, 256*1024, 512*1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]
    group_name = dist._get_process_group_name(dist.group.WORLD)

    if rank == 0:
        print(f"{'Size':>10} | {'NCCL (us)':>15} | {'One-Shot (us)':>15} | {'Speedup':>10}")
        print("-" * 60)

    for size in sizes_in_bytes:
        num_el = size // 4  # float32
        
        # 1. 准备 Standard NCCL 数据
        t_std = torch.randn(num_el, device=device)
        
        # 2. 准备 SymMem 数据
        t_sym = symm_mem.empty(num_el, dtype=torch.float32, device=device)
        t_sym.copy_(t_std)
        hdl = symm_mem.rendezvous(t_sym, dist.group.WORLD)
        
        # 定义测试闭包
        def run_nccl():
            dist.all_reduce(t_std, op=dist.ReduceOp.SUM)

        def run_oneshot():
            # 注意：one_shot_all_reduce 返回新 Tensor，我们这里测量算子发射耗时
            _ = torch.ops.symm_mem.one_shot_all_reduce(t_sym, "sum", group_name)

        # 执行测试
        dist.barrier()
        nccl_latency = benchmark_op(run_nccl)
        
        dist.barrier()
        oneshot_latency = benchmark_op(run_oneshot)

        if rank == 0:
            speedup = nccl_latency / oneshot_latency
            size_str = f"{size/1024:.0f}KB" if size < 1024*1024 else f"{size/1024/1024:.0f}MB"
            print(f"{size_str:>10} | {nccl_latency:>15.2f} | {oneshot_latency:>15.2f} | {speedup:>9.2f}x")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()