import os
import time
import torch
import torch.distributed as dist

# 假设你的环境中已经包含了该底层优化库的支持
try:
    from torch.distributed._symmetric_memory import enable_symm_mem_for_group
    HAS_SYMM_MEM = True
except ImportError:
    HAS_SYMM_MEM = False
    print("Warning: torch.distributed._symmetric_memory not found. Async-TP may fail.")

def benchmark_mlp_tp(B=4, S=32768, H=4096, iters=50, warmup=10):
    # ==========================================
    # 1. 分布式环境初始化
    # ==========================================
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 创建 TP 进程组
    tp_group = dist.new_group(backend="nccl")
    
    # 激活硬件级的对称内存单边通信 (One-sided DMA)
    if HAS_SYMM_MEM:
        enable_symm_mem_for_group(tp_group.group_name)

    # ==========================================
    # 2. 准备模型超参数与张量
    # 参考 LLaMA-7B 比例: 隐藏层维度 4096，MLP中间维度 11008
    # ==========================================
    H_inter = 11008
    
    assert S % world_size == 0, "Sequence length must be divisible by world_size"
    assert H_inter % world_size == 0, "Intermediate hidden size must be divisible by world_size"
    
    S_chunk = S // world_size
    H_inter_chunk = H_inter // world_size

    if local_rank == 0:
        print(f"{'='*60}")
        print(f"🚀 FFN/MLP TP Benchmark (SwiGLU)")
        print(f"📦 Config: B={B}, S={S}, H={H}, H_inter={H_inter}, GPUs={world_size}")
        print(f"{'='*60}")

    # 输入张量 (经过 Sequence Parallel 切分的状态)
    x_mlp_in = torch.randn(B, S_chunk, H, dtype=torch.bfloat16, device=device)

    # 权重矩阵
    # Column Parallel 权重 (按输出维度 H_inter 切分)
    w_gate = torch.randn(H, H_inter_chunk, dtype=torch.bfloat16, device=device)
    w_up   = torch.randn(H, H_inter_chunk, dtype=torch.bfloat16, device=device)
    
    # Row Parallel 权重 (按输入维度 H_inter 切分)
    w_down = torch.randn(H_inter_chunk, H, dtype=torch.bfloat16, device=device)

    # ==========================================
    # 模式 1: Standard TP (Megatron-LM Baseline)
    # ==========================================
    # (silu(x * W_gate) * (x * W_up))*W_down
    def run_standard_tp():
        # --- 阶段 A: All-Gather ---
        # 产生大量显存碎片的高危区：申请巨大的连续显存
        x_gathered = torch.empty(B, S, H, dtype=torch.bfloat16, device=device)
        dist.all_gather_into_tensor(x_gathered, x_mlp_in, group=tp_group) 
        
        # --- 阶段 B: 并行计算局部特征 ---
        gate_out = torch.matmul(x_gathered, w_gate)  # [B, S, H_inter_chunk]
        # gate_out = x_gathered * w_gate
        up_out   = torch.matmul(x_gathered, w_up)      # [B, S, H_inter_chunk]
        # up_out   = x_gathered * w_up
        
        # --- 阶段 C: 激活函数 (纯 Local) ---
        act_out = torch.nn.functional.silu(gate_out) * up_out
        
        # --- 阶段 D: 第二层投影求局部和 ---
        down_out = torch.matmul(act_out, w_down)     # [B, S, H]
        
        # --- 阶段 E: Reduce-Scatter ---
        y_scattered = torch.empty(B, S_chunk, H, dtype=torch.bfloat16, device=device)
        dist.reduce_scatter_tensor(y_scattered, down_out, group=tp_group)
        
        return y_scattered

    # ==========================================
    # 模式 2: Async-TP (Symmetric Memory)
    # ==========================================
    def run_symm_mem_tp():
        # --- 阶段 A: Fused AllGather + Matmul ---
        # 零额外缓冲分配，边拉数据边计算 SwiGLU 的两个分支
        _, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            x_mlp_in,
            [w_gate, w_up],
            gather_dim=1, # 沿序列维度 Gather
            group_name=tp_group.group_name,
        )
        
        # --- 阶段 B: 激活函数 ---
        act_out = torch.nn.functional.silu(mm_outputs[0]) * mm_outputs[1]

        # --- 阶段 C: Fused Matmul + ReduceScatter ---
        # 算完立刻 Push 到目标卡，无需完整中间结果的 buffer
        y_scattered = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            act_out,
            w_down,
            "sum",
            scatter_dim=1, # 沿序列维度切片
            group_name=tp_group.group_name,
        )
        
        return y_scattered

    # ==========================================
    # 压测执行框架
    # ==========================================
    def run_benchmark(func, name):
        # OOM 保护：测试前清空一下环境
        torch.cuda.empty_cache()
        
        # Warmup
        for _ in range(warmup):
            func()
        torch.cuda.synchronize()
        dist.barrier()
        
        # 抓取起始内存用于观察显存开销 (可选)
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.perf_counter()
        for _ in range(iters):
            func()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_time_ms = ((end_time - start_time) / iters) * 1000
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
        return avg_time_ms, peak_mem_gb

    # 开始压测
    if HAS_SYMM_MEM:
        std_time, std_mem = run_benchmark(run_standard_tp, "Standard TP")
        symm_time, symm_mem = run_benchmark(run_symm_mem_tp, "Symm-Mem TP")

        if local_rank == 0:
            speedup = (std_time - symm_time) / std_time * 100
            print(f"📊 [Standard TP] Time: {std_time:>8.2f} ms | Peak Mem: {std_mem:>6.2f} GB")
            print(f"🚀 [Async-TP]    Time: {symm_time:>8.2f} ms | Peak Mem: {symm_mem:>6.2f} GB")
            print(f"{'-'*60}")
            print(f"⚡ Time Saved  : {std_time - symm_time:>8.2f} ms")
            print(f"📈 Speedup     : +{speedup:>8.2f} %")
            print(f"📉 Mem Saved   : {std_mem - symm_mem:>8.2f} GB")
            print(f"{'='*60}")
    else:
        if local_rank == 0:
            print("由于缺失对称内存算子，仅运行 Baseline...")
        std_time, std_mem = run_benchmark(run_standard_tp, "Standard TP")
        if local_rank == 0:
            print(f"📊 [Standard TP] Time: {std_time:>8.2f} ms | Peak Mem: {std_mem:>6.2f} GB")

    dist.destroy_process_group()

if __name__ == "__main__":
    benchmark_mlp_tp()