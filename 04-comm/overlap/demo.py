import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
import time
import os

def benchmark_symm_mem_tp(B=4, S=32728, H=4096, iters=50, warmup=10):
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    # 1. 创建 TP 进程组并激活 Symmetric Memory
    tp_group = dist.new_group(backend="nccl")
    # 这句话是开启底层硬件单边通信 (One-sided DMA) 的钥匙
    enable_symm_mem_for_group(tp_group.group_name)

    # 2. 准备张量形状与数据 (结合 Sequence Parallelism)
    S_chunk = S // world_size  # 每个卡分到的 Sequence 长度
    
    # 模拟输入 (QKV 层的前置输入)
    x = torch.randn(B, S_chunk, H, dtype=torch.bfloat16, device=device)
    # QKV 的权重 (Column Parallel, 沿着输出维度切分)
    H_out_chunk = H // world_size
    wq = torch.randn(H, H_out_chunk, dtype=torch.bfloat16, device=device)
    wk = torch.randn(H, H_out_chunk, dtype=torch.bfloat16, device=device)
    wv = torch.randn(H, H_out_chunk, dtype=torch.bfloat16, device=device)

    # 模拟输出投影层的输入 (O_proj)
    # 输入是完整的 Sequence，特征维度是被切分的
    x_out = torch.randn(B, S, H_out_chunk, dtype=torch.bfloat16, device=device)
    # O_proj 的权重 (Row Parallel, 沿着输入维度切分)
    w_out = torch.randn(H_out_chunk, H, dtype=torch.bfloat16, device=device)

    # ==========================================
    # 模式 1: 传统的 TP (分离的通信与计算)
    # ==========================================
    # x_out [B, S, H_out_chunk]
    # w_out [H_out_chunk, H]
    def run_standard_tp():
        # --- QKV 阶段 ---
        # 1. All-Gather
        x_gathered = torch.empty(B, S, H, dtype=torch.bfloat16, device=device)
        dist.all_gather_into_tensor(x_gathered, x, group=tp_group)
        # 2. 计算 Matmul
        q = torch.matmul(x_gathered, wq)
        k = torch.matmul(x_gathered, wk)
        v = torch.matmul(x_gathered, wv)

        # --- O_proj 阶段 ---
        # 1. 计算 Matmul
        y = torch.matmul(x_out, w_out)
        # 2. Reduce-Scatter
        y_scattered = torch.empty(B, S_chunk, H, dtype=torch.bfloat16, device=device)
        dist.reduce_scatter_tensor(y_scattered, y, group=tp_group)
        
        return q, k, v, y_scattered

    # ==========================================
    # 模式 2: Fused Async-TP (对称内存 API)
    # ==========================================
    def run_symm_mem_tp():
        # --- QKV 阶段 (All-Gather -> Matmul) ---
        # 硬件会在拉取相邻卡数据的同时，把已经拉到的部分送进 Tensor Core
        ag_out, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            x,
            [wq, wk, wv],
            gather_dim=1,
            group_name=tp_group.group_name,
        )
        
        # --- O_proj 阶段 (Matmul -> Reduce-Scatter) ---
        # Tensor Core 算完一个分块，NVLink 立刻将其推 (Push) 到目标卡的显存中
        rs_out = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            x_out,
            w_out,
            "sum", # 注意这里是 sum 累加，大模型通常用 sum
            scatter_dim=1,
            group_name=tp_group.group_name,
        )
        
        return mm_outputs[0], mm_outputs[1], mm_outputs[2], rs_out

    # ==========================================
    # 压测工具
    # ==========================================
    def benchmark(func, name):
        # 预热
        for _ in range(warmup):
            func()
        torch.cuda.synchronize()
        
        dist.barrier()
        start = time.perf_counter()
        for _ in range(iters):
            func()
        torch.cuda.synchronize()
        end = time.perf_counter()
        return ((end - start) / iters) * 1000

    if local_rank == 0:
        print(f"{'='*60}")
        print(f"Matrix Size: B={B}, S={S}, H={H} | Precision: BF16")
        print(f"{'='*60}")

    # 执行压测
    time_std = benchmark(run_standard_tp, "Standard TP")
    time_symm = benchmark(run_symm_mem_tp, "Symmetric Memory TP")

    if local_rank == 0:
        speedup = (time_std - time_symm) / time_std * 100
        print(f"Standard TP Time : {time_std:>8.2f} ms")
        print(f"Symm-Mem TP Time : {time_symm:>8.2f} ms")
        print(f"Time Saved       : {time_std - time_symm:>8.2f} ms")
        print(f"Performance Gain : +{speedup:>8.2f} %")
        print(f"{'='*60}")

    dist.destroy_process_group()

if __name__ == "__main__":
    benchmark_symm_mem_tp()