import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from tensor_parallel_layers import ColumnParallelLinear, RowParallelLinear

def demo_tp(rank, world_size):
    # ---------------------------------------------------
    # 1. 环境初始化
    # ---------------------------------------------------
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 使用 NCCL 后端 (GPU 通信标准)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    print(f"[Rank {rank}] Process initialized on GPU {rank}")

    # ---------------------------------------------------
    # 2. 定义模型参数
    # ---------------------------------------------------
    batch_size = 2
    seq_len = 4
    hidden_size = 8
    intermediate_size = 16  # MLP 中间层放大维度
    
    # ---------------------------------------------------
    # 3. 准备权重 (Golden Weights)
    # 为了验证正确性，我们在 Rank 0 创建一套完整的权重，并广播给所有 Rank
    # ---------------------------------------------------
    torch.manual_seed(42)
    # W1: [inter, hidden], B1: [inter]
    full_w1 = torch.randn(intermediate_size, hidden_size).cuda(rank)
    full_b1 = torch.randn(intermediate_size).cuda(rank)
    # W2: [hidden, inter], B2: [hidden]
    full_w2 = torch.randn(hidden_size, intermediate_size).cuda(rank)
    full_b2 = torch.randn(hidden_size).cuda(rank)
    
    # 确保所有显卡用的是同一套随机权重
    dist.broadcast(full_w1, src=0)
    dist.broadcast(full_b1, src=0)
    dist.broadcast(full_w2, src=0)
    dist.broadcast(full_b2, src=0)

    # ---------------------------------------------------
    # 4. 构建 TP 模型
    # ---------------------------------------------------
    # 第一层：Column Parallel (切分输出维度 intermediate_size)
    col_layer = ColumnParallelLinear(hidden_size, intermediate_size, bias=True)
    # 加载权重 (内部会自动根据 Rank 切片)
    col_layer.weight_loader(col_layer.weight, full_w1)
    col_layer.bias_loader(col_layer.bias, full_b1)
    
    # 第二层：Row Parallel (切分输入维度 intermediate_size)
    row_layer = RowParallelLinear(intermediate_size, hidden_size, bias=True)
    # 加载权重
    row_layer.weight_loader(row_layer.weight, full_w2)
    row_layer.bias_loader(row_layer.bias, full_b2)
    
    # 放到 GPU
    model = nn.Sequential(col_layer, nn.ReLU(), row_layer).cuda(rank)
    
    # ---------------------------------------------------
    # 5. 前向传播
    # ---------------------------------------------------
    # 构造输入 (所有 Rank 必须持有相同的输入)
    torch.manual_seed(123)
    x = torch.randn(batch_size, seq_len, hidden_size).cuda(rank)
    dist.broadcast(x, src=0) # 确保输入一致
    
    # TP Forward
    y_tp = model(x)
    
    # ---------------------------------------------------
    # 6. 验证正确性
    # ---------------------------------------------------
    if rank == 0:
        print(f"\n[Rank {rank}] --- Verification ---")
        print(f"[Rank {rank}] Input Shape: {x.shape}")
        print(f"[Rank {rank}] Output Shape: {y_tp.shape}")
        
        # 本地单卡计算标准结果 (Reference)
        # Standard Linear: X @ W.T + b
        # Layer 1
        y_ref = F.linear(x, full_w1, full_b1)
        y_ref = F.relu(y_ref)
        # Layer 2
        y_ref = F.linear(y_ref, full_w2, full_b2)
        
        # 比较误差
        diff = (y_tp - y_ref).abs().max()
        print(f"[Rank {rank}] Max Error: {diff.item():.8f}")
        
        if diff < 1e-5:
            print(f"[Rank {rank}] ✅ SUCCESS: TP result matches single-GPU result!")
        else:
            print(f"[Rank {rank}] ❌ FAILED: Results mismatch.")

    # ---------------------------------------------------
    # 7. 清理
    # ---------------------------------------------------
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # 检查环境是否有足够的 GPU
    if torch.cuda.device_count() < world_size:
        print(f"Error: This demo requires at least {world_size} GPUs.")
    else:
        print(f"Starting {world_size}-GPU Tensor Parallelism Demo...")
        mp.spawn(demo_tp, args=(world_size,), nprocs=world_size, join=True)