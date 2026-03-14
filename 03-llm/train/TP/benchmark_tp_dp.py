"""
TP (Tensor Parallelism) vs DP (Data Parallelism) vs 单卡性能对比

核心差异分析：

1. 模型存储：
   - 单卡：完整模型
   - DP：每卡完整模型（冗余存储）
   - TP：每卡 1/N 模型（切分存储）

2. 通信模式：
   - 单卡：无通信
   - DP：反向传播后 all-reduce 梯度（通信量 = 模型参数量）
   - TP：前向传播中 all-reduce/all-gather（通信量 = 激活值大小）

3. 适用场景：
   - 单卡：小模型、小批量
   - DP：模型能放入单卡、需要大批量
   - TP：模型无法放入单卡、超大模型

运行方式：
  # 单卡基准
  python benchmark_tp_dp.py --mode single

  # DP 性能测试 (需要多卡)
  torchrun --nproc_per_node=2 benchmark_tp_dp.py --mode dp

  # TP 性能测试 (需要多卡)
  torchrun --nproc_per_node=2 benchmark_tp_dp.py --mode tp

  # 全面对比 (需要多卡)
  torchrun --nproc_per_node=2 benchmark_tp_dp.py --mode compare
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
import argparse
import time
import math
from dataclasses import dataclass
from typing import Optional


# ============== 通信原语 (复用自 tp.py) ==============

def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def all_reduce(tensor, op=dist.ReduceOp.SUM):
    if get_world_size() == 1:
        return tensor
    dist.all_reduce(tensor, op=op)
    return tensor


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dim=-1):
        ctx.dim = dim
        world_size = get_world_size()
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        world_size = get_world_size()
        rank = get_rank()
        chunks = torch.chunk(grad_output, world_size, dim=dim)
        return chunks[rank].contiguous(), None


def all_gather(tensor, dim=-1):
    if get_world_size() == 1:
        return tensor
    return _AllGather.apply(tensor, dim)


# ============== TP MLP (简化版，来自 tp.py) ==============

class ColumnParallelLinear(nn.Module):
    """列并行线性层：按输出维度切分"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        world_size = get_world_size()
        self.world_size = world_size
        self.out_features_per_gpu = out_features // world_size

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_gpu, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_gpu))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))

    def forward(self, x):
        output = x @ self.weight.t()
        if self.bias is not None:
            output = output + self.bias
        return output


class RowParallelLinear(nn.Module):
    """行并行线性层：按输入维度切分"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        world_size = get_world_size()
        self.world_size = world_size
        self.in_features_per_gpu = in_features // world_size

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_gpu)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))

    def forward(self, x):
        output = x @ self.weight.t()
        if self.world_size > 1:
            all_reduce(output)
        if self.bias is not None:
            output = output + self.bias
        return output


class TPMLP(nn.Module):
    """TP MLP: 列并行 -> 行并行（高效组合）"""

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, bias=True)
        self.fc2 = RowParallelLinear(hidden_features, out_features, bias=True)

    def forward(self, x):
        h = self.fc1(x)
        h = F.gelu(h)
        out = self.fc2(h)
        return out


# ============== 标准 MLP (用于单卡和 DP) ==============

class StandardMLP(nn.Module):
    """标准 MLP"""

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        h = self.fc1(x)
        h = F.gelu(h)
        out = self.fc2(h)
        return out


# ============== 性能测试结果 ==============

@dataclass
class BenchmarkResult:
    mode: str
    total_time: float
    forward_time: float
    backward_time: float
    comm_time: float
    memory_allocated: float
    memory_reserved: float
    num_params: int
    batch_size: int
    num_iterations: int

    def __str__(self):
        return f"""
{'=' * 60}
{self.mode} 性能报告
{'=' * 60}
总时间:           {self.total_time:.4f}s ({self.total_time/self.num_iterations*1000:.2f}ms/iter)
前向传播时间:     {self.forward_time:.4f}s ({self.forward_time/self.num_iterations*1000:.2f}ms/iter)
反向传播时间:     {self.backward_time:.4f}s ({self.backward_time/self.num_iterations*1000:.2f}ms/iter)
通信时间:         {self.comm_time:.4f}s ({self.comm_time/self.num_iterations*1000:.2f}ms/iter)
GPU 显存占用:     {self.memory_allocated:.2f} MB (峰值: {self.memory_reserved:.2f} MB)
参数量:           {self.num_params:,}
Batch Size:       {self.batch_size}
迭代次数:         {self.num_iterations}
{'=' * 60}
"""


# ============== 基准测试函数 ==============

def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' not in os.environ:
        return 0, 1

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl')
    return rank, world_size


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_model_size(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def benchmark_single_gpu(args):
    """单卡性能测试"""
    device = 'cuda'
    torch.manual_seed(42)

    model = StandardMLP(args.in_features, args.hidden_features, args.out_features).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # 准备数据
    x = torch.randn(args.batch_size, args.in_features).to(device)
    y = torch.randint(0, args.out_features, (args.batch_size,)).to(device)

    # 预热
    for _ in range(args.warmup):
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # 正式测试
    forward_time = 0
    backward_time = 0
    total_start = time.time()

    for _ in range(args.iterations):
        optimizer.zero_grad()

        # 前向传播
        torch.cuda.synchronize()
        fwd_start = time.time()
        output = model(x)
        loss = F.cross_entropy(output, y)
        torch.cuda.synchronize()
        forward_time += time.time() - fwd_start

        # 反向传播
        torch.cuda.synchronize()
        bwd_start = time.time()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        backward_time += time.time() - bwd_start

    total_time = time.time() - total_start

    memory_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
    memory_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024

    return BenchmarkResult(
        mode="单卡",
        total_time=total_time,
        forward_time=forward_time,
        backward_time=backward_time,
        comm_time=0,
        memory_allocated=memory_allocated,
        memory_reserved=memory_reserved,
        num_params=get_model_size(model),
        batch_size=args.batch_size,
        num_iterations=args.iterations
    )


def benchmark_dp(rank, world_size, args):
    """DP 性能测试"""
    device = torch.device(f'cuda:{rank}')
    torch.manual_seed(42)

    model = StandardMLP(args.in_features, args.hidden_features, args.out_features).to(device)

    # 同步初始权重
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # 每个进程处理部分数据
    local_batch_size = args.batch_size // world_size
    x = torch.randn(local_batch_size, args.in_features).to(device)
    y = torch.randint(0, args.out_features, (local_batch_size,)).to(device)

    # 预热
    for _ in range(args.warmup):
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        # DP: all-reduce 梯度
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)
        optimizer.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # 正式测试
    forward_time = 0
    backward_time = 0
    comm_time = 0
    total_start = time.time()

    for _ in range(args.iterations):
        optimizer.zero_grad()

        # 前向传播
        torch.cuda.synchronize()
        fwd_start = time.time()
        output = model(x)
        loss = F.cross_entropy(output, y)
        torch.cuda.synchronize()
        forward_time += time.time() - fwd_start

        # 反向传播
        torch.cuda.synchronize()
        bwd_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_time += time.time() - bwd_start

        # 通信
        torch.cuda.synchronize()
        comm_start = time.time()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)
        torch.cuda.synchronize()
        comm_time += time.time() - comm_start

        optimizer.step()

    total_time = time.time() - total_start

    memory_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
    memory_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024

    return BenchmarkResult(
        mode=f"DP (world_size={world_size})",
        total_time=total_time,
        forward_time=forward_time,
        backward_time=backward_time,
        comm_time=comm_time,
        memory_allocated=memory_allocated,
        memory_reserved=memory_reserved,
        num_params=get_model_size(model),
        batch_size=args.batch_size,
        num_iterations=args.iterations
    )


def benchmark_tp(rank, world_size, args):
    """TP 性能测试"""
    device = torch.device(f'cuda:{rank}')
    torch.manual_seed(42)

    # TP 模型：每个 GPU 只存储 1/world_size 的权重
    model = TPMLP(args.in_features, args.hidden_features, args.out_features).to(device)

    # 同步初始权重
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # TP: 输入是完整的（列并行需要完整输入）
    x = torch.randn(args.batch_size, args.in_features).to(device)
    y = torch.randint(0, args.out_features, (args.batch_size,)).to(device)

    # 预热
    for _ in range(args.warmup):
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        # TP: 部分梯度需要同步（在 autograd 中已处理，但需要同步权重梯度）
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)
        optimizer.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # 正式测试
    forward_time = 0
    backward_time = 0
    comm_time = 0
    total_start = time.time()

    for _ in range(args.iterations):
        optimizer.zero_grad()

        # 前向传播（包含 TP 通信）
        torch.cuda.synchronize()
        fwd_start = time.time()
        output = model(x)
        loss = F.cross_entropy(output, y)
        torch.cuda.synchronize()
        forward_time += time.time() - fwd_start

        # 反向传播
        torch.cuda.synchronize()
        bwd_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_time += time.time() - bwd_start

        # 梯度同步
        torch.cuda.synchronize()
        comm_start = time.time()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)
        torch.cuda.synchronize()
        comm_time += time.time() - comm_start

        optimizer.step()

    total_time = time.time() - total_start

    memory_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
    memory_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024

    # TP 每卡参数量
    tp_params = get_model_size(model)

    return BenchmarkResult(
        mode=f"TP (world_size={world_size})",
        total_time=total_time,
        forward_time=forward_time,
        backward_time=backward_time,
        comm_time=comm_time,
        memory_allocated=memory_allocated,
        memory_reserved=memory_reserved,
        num_params=tp_params * world_size,  # 总参数量
        batch_size=args.batch_size,
        num_iterations=args.iterations
    )


def print_comparison(results: list[BenchmarkResult], args):
    """打印对比报告"""
    print("\n" + "=" * 70)
    print("TP vs DP vs 单卡 性能对比报告")
    print("=" * 70)
    print(f"模型配置: input={args.in_features}, hidden={args.hidden_features}, output={args.out_features}")
    print(f"Batch Size: {args.batch_size}, Iterations: {args.iterations}")
    print("=" * 70)

    # 表头
    print(f"\n{'指标':<20} {'单卡':>15} {'DP':>15} {'TP':>15}")
    print("-" * 70)

    # 找到各模式的结果
    single_result = next((r for r in results if "单卡" in r.mode), None)
    dp_result = next((r for r in results if "DP" in r.mode), None)
    tp_result = next((r for r in results if "TP" in r.mode), None)

    # 每迭代时间
    print(f"{'每迭代时间 (ms)':<20} ", end="")
    if single_result:
        print(f"{single_result.total_time/single_result.num_iterations*1000:>15.2f} ", end="")
    else:
        print(f"{'N/A':>15} ", end="")
    if dp_result:
        print(f"{dp_result.total_time/dp_result.num_iterations*1000:>15.2f} ", end="")
    else:
        print(f"{'N/A':>15} ", end="")
    if tp_result:
        print(f"{tp_result.total_time/tp_result.num_iterations*1000:>15.2f}")
    else:
        print(f"{'N/A':>15}")

    # 前向时间
    print(f"{'前向时间 (ms)':<20} ", end="")
    if single_result:
        print(f"{single_result.forward_time/single_result.num_iterations*1000:>15.2f} ", end="")
    else:
        print(f"{'N/A':>15} ", end="")
    if dp_result:
        print(f"{dp_result.forward_time/dp_result.num_iterations*1000:>15.2f} ", end="")
    else:
        print(f"{'N/A':>15} ", end="")
    if tp_result:
        print(f"{tp_result.forward_time/tp_result.num_iterations*1000:>15.2f}")
    else:
        print(f"{'N/A':>15}")

    # 反向时间
    print(f"{'反向时间 (ms)':<20} ", end="")
    if single_result:
        print(f"{single_result.backward_time/single_result.num_iterations*1000:>15.2f} ", end="")
    else:
        print(f"{'N/A':>15} ", end="")
    if dp_result:
        print(f"{dp_result.backward_time/dp_result.num_iterations*1000:>15.2f} ", end="")
    else:
        print(f"{'N/A':>15} ", end="")
    if tp_result:
        print(f"{tp_result.backward_time/tp_result.num_iterations*1000:>15.2f}")
    else:
        print(f"{'N/A':>15}")

    # 通信时间
    print(f"{'通信时间 (ms)':<20} ", end="")
    if single_result:
        print(f"{0:>15.2f} ", end="")
    else:
        print(f"{'N/A':>15} ", end="")
    if dp_result:
        print(f"{dp_result.comm_time/dp_result.num_iterations*1000:>15.2f} ", end="")
    else:
        print(f"{'N/A':>15} ", end="")
    if tp_result:
        print(f"{tp_result.comm_time/tp_result.num_iterations*1000:>15.2f}")
    else:
        print(f"{'N/A':>15}")

    # 显存占用
    print(f"{'显存占用 (MB)':<20} ", end="")
    if single_result:
        print(f"{single_result.memory_allocated:>15.2f} ", end="")
    else:
        print(f"{'N/A':>15} ", end="")
    if dp_result:
        print(f"{dp_result.memory_allocated:>15.2f} ", end="")
    else:
        print(f"{'N/A':>15} ", end="")
    if tp_result:
        print(f"{tp_result.memory_allocated:>15.2f}")
    else:
        print(f"{'N/A':>15}")

    # 参数量（每卡）
    print(f"{'每卡参数量':<20} ", end="")
    if single_result:
        print(f"{single_result.num_params:>15,} ", end="")
    else:
        print(f"{'N/A':>15} ", end="")
    if dp_result:
        print(f"{dp_result.num_params:>15,} ", end="")
    else:
        print(f"{'N/A':>15} ", end="")
    if tp_result:
        print(f"{tp_result.num_params//int(tp_result.mode.split('=')[1].rstrip(')')):>15,}")
    else:
        print(f"{'N/A':>15}")

    print("-" * 70)

    # 分析总结
    print("\n分析总结:")
    print("-" * 70)

    if dp_result and single_result:
        speedup = single_result.total_time / dp_result.total_time
        print(f"  DP 相比单卡加速比: {speedup:.2f}x")
        print(f"  DP 通信开销占比: {dp_result.comm_time/dp_result.total_time*100:.1f}%")
        print(f"  DP 显存节省: {(1 - dp_result.memory_allocated/single_result.memory_allocated)*100:.1f}% (无)")

    if tp_result and single_result:
        speedup = single_result.total_time / tp_result.total_time
        print(f"  TP 相比单卡加速比: {speedup:.2f}x")
        print(f"  TP 通信开销占比: {tp_result.comm_time/tp_result.total_time*100:.1f}%")
        tp_world_size = int(tp_result.mode.split('=')[1].rstrip(')'))
        print(f"  TP 显存节省: {(1 - tp_result.memory_allocated/single_result.memory_allocated)*100:.1f}%")
        print(f"  TP 每卡参数量: 单卡的 {1/tp_world_size*100:.1f}%")

    print("\n关键结论:")
    print("-" * 70)
    print("""
  1. 显存占用:
     - 单卡和DP: 每卡存储完整模型
     - TP: 每卡只存储 1/N 模型，适合超大模型

  2. 通信模式:
     - DP: 反向传播后 all-reduce 梯度，通信量 = 参数量
     - TP: 前向传播中 all-reduce 激活值，通信量 = 激活值大小

  3. 适用场景:
     - 单卡: 小模型、快速原型
     - DP: 模型能放入单卡、需要增大有效 batch size
     - TP: 模型太大无法放入单卡、超大模型训练

  4. 性能权衡:
     - DP: 通信与计算可重叠，适合计算密集型
     - TP: 通信在关键路径上，对通信带宽敏感
""")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='TP vs DP vs Single GPU Benchmark')
    parser.add_argument('--mode', type=str, default='compare',
                        choices=['single', 'dp', 'tp', 'compare'],
                        help='Running mode')
    parser.add_argument('--in_features', type=int, default=1024, help='Input features')
    parser.add_argument('--hidden_features', type=int, default=4096, help='Hidden features')
    parser.add_argument('--out_features', type=int, default=1024, help='Output features')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    args = parser.parse_args()

    rank, world_size = setup_distributed()

    results = []

    if args.mode == 'single':
        if rank == 0:
            print("\n运行单卡基准测试...")
            result = benchmark_single_gpu(args)
            print(result)

    elif args.mode == 'dp':
        if world_size < 2:
            print("DP 需要至少 2 个 GPU")
            return
        if rank == 0:
            print(f"\n运行 DP 基准测试 (world_size={world_size})...")
        result = benchmark_dp(rank, world_size, args)
        if rank == 0:
            print(result)

    elif args.mode == 'tp':
        if world_size < 2:
            print("TP 需要至少 2 个 GPU")
            return
        if rank == 0:
            print(f"\n运行 TP 基准测试 (world_size={world_size})...")
        result = benchmark_tp(rank, world_size, args)
        if rank == 0:
            print(result)

    elif args.mode == 'compare':
        # 综合对比
        if rank == 0:
            print("\n" + "=" * 70)
            print("TP vs DP vs 单卡 综合性能对比")
            print("=" * 70)

        # 单卡测试
        if rank == 0:
            print("\n[1/3] 单卡基准测试...")
            single_result = benchmark_single_gpu(args)
            results.append(single_result)

        if world_size >= 2:
            # DP 测试
            if rank == 0:
                print("\n[2/3] DP 基准测试...")
            dp_result = benchmark_dp(rank, world_size, args)
            results.append(dp_result)

            # TP 测试
            if rank == 0:
                print("\n[3/3] TP 基准测试...")
            tp_result = benchmark_tp(rank, world_size, args)
            results.append(tp_result)
        else:
            print("\n[跳过] DP 和 TP 需要至少 2 个 GPU")

        if rank == 0:
            print_comparison(results, args)

    cleanup_distributed()


if __name__ == "__main__":
    main()