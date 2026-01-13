import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

def setup(rank, world_size):
    """初始化分布式环境 (NCCL 后端是 GPU 通信的标准)"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 初始化进程组，使用 nccl 后端（GPU 之间通信最快）
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_expert_parallel(rank, world_size):
    setup(rank, world_size)
    
    # ==========================================
    # 0. 模拟设置
    # ==========================================
    # 假设每张卡负责 1 个专家 (简化场景)
    # Rank 0 -> Expert 0
    # Rank 1 -> Expert 1
    device = torch.device(f"cuda:{rank}")
    
    # 模拟输入: [Batch=2, Seq=4, Dim=8]
    # 每个 GPU 都有自己的一批输入数据
    local_batch_size = 2
    seq_len = 4
    dim = 8
    
    # 生成一些随机数据
    local_tokens = torch.randn(local_batch_size * seq_len, dim).to(device)
    
    # ==========================================
    # 1. 路由 (Gating)
    # ==========================================
    # 简单模拟：随机决定每个 Token 去哪个 GPU (Expert)
    # 0 代表去 Rank 0, 1 代表去 Rank 1
    router_logits = torch.randn(local_tokens.size(0), world_size).to(device)
    target_ranks = torch.argmax(router_logits, dim=1) # [8]
    
    # 打印路由信息
    print(f"[GPU {rank}] Token 目标分布: {target_ranks.tolist()}")

    # ========================================== 
    # 2. 准备发送数据 (Sort & Pack)
    # ==========================================
    # 为了使用 all_to_all，我们需要把发往 Rank 0 的放前面，Rank 1 的放后面
    sorted_indices = torch.argsort(target_ranks)
    sorted_tokens = local_tokens[sorted_indices]
    
    # 计算我要发给每个 Rank 多少个 Token
    # output_splits 列表: [发给Rank0的数量, 发给Rank1的数量]
    output_splits = torch.bincount(target_ranks, minlength=world_size).tolist()
    
    print(f"[GPU {rank}] 准备发送: {output_splits} (总数 {sum(output_splits)})")

    # ==========================================
    # 3. 交换元数据 (Exchange Metadata)
    # ==========================================
    # 在发送真正的 Token 之前，我们需要知道“别人要发给我多少个 Token”
    # 这样我才能准备好接收的 Buffer
    
    # input_splits: 用来存储别人发给我的数量
    output_splits_tensor = torch.tensor(output_splits, device=device)
    input_splits_tensor = torch.zeros(world_size, dtype=torch.long, device=device)
    
    # All-to-All 交换数字：Rank i 告诉 Rank j "我有多少数据给你"
    dist.all_to_all_single(input_splits_tensor, output_splits_tensor)
    
    input_splits = input_splits_tensor.tolist()
    print(f"[GPU {rank}] 即将接收: {input_splits} (来自各 Rank)")

    # ==========================================
    # 4. 核心：Dispatch (All-to-All 传输 Token)
    # ==========================================
    # 准备接收容器
    total_tokens_to_receive = sum(input_splits)
    received_tokens = torch.zeros(total_tokens_to_receive, dim, device=device)
    
    # 执行物理传输！
    # 这一步会通过 NVLink/PCIe 把数据搬运到对应的 GPU
    dist.all_to_all_single(
        received_tokens,       # 接收 Buffer
        sorted_tokens,         # 发送 Buffer
        output_split_sizes=input_splits,  # 接收切分 (注意这里参数名的反直觉性，但在 single API 中由 splits 决定)
        input_split_sizes=output_splits   # 发送切分
    )
    
    # ==========================================
    # 5. 专家计算 (Expert Computation)
    # ==========================================
    # 此时，received_tokens 里的所有数据都是属于我(当前 GPU 负责的专家)的
    # 模拟 FFN 计算: x * 10 + rank
    # 这样我们可以通过数值验证数据是不是真的在这个 GPU 上算过
    expert_output = received_tokens * 10 + rank

    # ==========================================
    # 6. 核心：Combine (All-to-All 返回结果)
    # ==========================================
    # 我们要把算好的结果还给原来的 GPU
    # 流程和 Dispatch 刚好相反
    
    # 接收容器：大小应该和最开始发出去的一样
    final_output_sorted = torch.zeros_like(local_tokens)
    
    # 再次物理传输
    dist.all_to_all_single(
        final_output_sorted,
        expert_output,
        output_split_sizes=output_splits, # 这次我收回的量 = 上次我发出的量
        input_split_sizes=input_splits    # 这次我发出的量 = 上次我收到的量
    )
    
    # 最后一步：把顺序还原 (Unsort)
    # 这里的逻辑稍微复杂一点，为简化 demo，我们只打印前几个值
    # 实际上需要用 inverse permutation 还原顺序
    
    # 验证：打印第一个 Token 的值
    # 如果它去了 Rank 1，值应该是 (原始值 * 10 + 1)
    # 如果它去了 Rank 0，值应该是 (原始值 * 10 + 0)
    print(f"[GPU {rank}] 完成! 采样数据: {final_output_sorted[0, :2].tolist()}")

    cleanup()

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(f"需要至少 2 个 GPU 才能运行此 demo，当前检测到 {n_gpus} 个。")
    else:
        world_size = 2 # 只使用前两张卡
        print(f"启动 {world_size} 个进程进行专家并行演示...")
        mp.spawn(run_expert_parallel,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)