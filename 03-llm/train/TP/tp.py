"""
张量并行 (Tensor Parallelism) 实现
实现 MLP 层的列并行和行并行

核心思想：
1. 列并行 (Column Parallel): 将线性层的权重按列切分，每个 GPU 持有部分输出维度
   - 前向：X @ W_i (W_i 是 W 的第 i 切分)，输出需要在 hidden 维度上 all-gather
   - 反向：对输入梯度 all-reduce

2. 行并行 (Row Parallel): 将线性层的权重按行切分，每个 GPU 持有部分输入维度
   - 前向：X_i @ W_i (X_i 是 X 的第 i 切分)，输出需要 all-reduce
   - 反向：对输入梯度 all-gather

典型组合：列并行 -> 行并行，可以避免中间通信
  - 列并行输出在 hidden 维度上切分，直接作为行并行的切分输入
  - 只需要在最后做一次 all-reduce

运行方式：
  torchrun --nproc_per_node=2 tp.py --mode verify
  torchrun --nproc_per_node=2 tp.py --mode train
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
import argparse
import math


# ============== 通信原语 ==============

def get_world_size():
    """获取进程数"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    """获取当前进程编号"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def all_reduce(tensor, op=dist.ReduceOp.SUM):
    """
    All-reduce: 所有进程的 tensor 求和，结果广播到所有进程
    用于行并行的前向传播
    """
    if get_world_size() == 1:
        return tensor
    dist.all_reduce(tensor, op=op)
    return tensor


class _AllGather(torch.autograd.Function):
    """可导的 All-Gather 操作"""
    @staticmethod
    def forward(ctx, tensor, dim=-1):
        ctx.dim = dim
        world_size = get_world_size()
        
        # 创建接收张量 (不需要梯度)
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        
        # 拼接并返回
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        """
        all_gather 的反向传播是 reduce_scatter 或者直接截取。
        因为每个 rank 只需要负责自己那部分的梯度。
        """
        dim = ctx.dim
        world_size = get_world_size()
        rank = get_rank()
        
        # 沿着拼接的维度把传回来的梯度切开
        chunks = torch.chunk(grad_output, world_size, dim=dim)
        
        # 返回属于当前进程的梯度，其余维度返回 None (对应 forward 中的 dim 参数)
        return chunks[rank].contiguous(), None


def all_gather(tensor, dim=-1):
    """
    All-gather: 收集所有进程的 tensor 并在指定维度上拼接
    支持 autograd 自动求导
    """
    if get_world_size() == 1:
        return tensor
        
    return _AllGather.apply(tensor, dim)

def reduce_scatter(tensor, dim=-1):
    """
    Reduce-scatter: 先 all-reduce 求和，然后每个进程获取对应切分的结果
    用于行并行的反向传播
    """
    world_size = get_world_size()
    if world_size == 1:
        return tensor

    # 先 all-reduce
    all_reduce(tensor)

    # 计算每个进程应该获取的切分
    split_size = tensor.shape[dim] // world_size
    chunks = torch.chunk(tensor, world_size, dim=dim)

    return chunks[get_rank()].contiguous()


# ============== 列并行线性层 ==============

class ColumnParallelLinearFunction(torch.autograd.Function):
    """
    列并行线性层的前向和反向传播
    将权重按列切分，每个 GPU 持有输出维度的一部分

    前向：X @ W_i + b_i (本地计算，输出是完整输出的第 i 切分)
    反向：需要对输入梯度进行 all-reduce
    """

    @staticmethod
    def forward(ctx, x, weight, bias, world_size):
        """
        前向传播：
        - x: [batch_size, in_features] 完整输入
        - weight: [out_features_per_gpu, in_features] 本地权重切分
        - bias: [out_features_per_gpu] 本地偏置切分
        - 输出: [batch_size, out_features_per_gpu] 本地输出切分
        """
        ctx.save_for_backward(x, weight, bias)
        ctx.world_size = world_size

        # 本地矩阵乘法
        output = x @ weight.t() + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：
        - grad_output: [batch_size, out_features_per_gpu] 本地输出梯度
        - 需要对输入梯度 all-reduce，因为每个 GPU 都需要完整的输入梯度
        """
        x, weight, bias = ctx.saved_tensors
        world_size = ctx.world_size

        # 计算本地梯度
        grad_x = grad_output @ weight
        grad_weight = grad_output.t() @ x
        grad_bias = grad_output.sum(dim=0)

        # 对输入梯度 all-reduce
        if world_size > 1:
            all_reduce(grad_x)

        return grad_x, grad_weight, grad_bias, None


class ColumnParallelLinear(nn.Module):
    """
    列并行线性层
    将权重按列切分，每个 GPU 持有输出维度的一部分

    特点：
    - 输入是完整的
    - 输出是切分的（每个 GPU 有部分输出维度）
    - 需要后续 all-gather 得到完整输出（或直接传给行并行层）
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        world_size = get_world_size()
        self.world_size = world_size

        # 每个进程持有 out_features // world_size 的输出维度
        assert out_features % world_size == 0, \
            f"out_features ({out_features}) must be divisible by world_size ({world_size})"

        self.out_features_per_gpu = out_features // world_size

        # 本地权重切分
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_gpu, in_features)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_gpu))
        else:
            self.register_parameter('bias', None)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        前向传播
        x: [batch_size, in_features] 完整输入
        输出: [batch_size, out_features_per_gpu] 本地输出切分
        """
        return ColumnParallelLinearFunction.apply(
            x, self.weight, self.bias, self.world_size
        )


# ============== 行并行线性层 ==============

class RowParallelLinearFunction(torch.autograd.Function):
    """
    行并行线性层的前向和反向传播
    将权重按行切分，每个 GPU 持有输入维度的一部分

    前向：X_i @ W_i (本地计算)，然后 all-reduce 得到完整输出
    反向：需要对输入梯度 all-gather
    """

    @staticmethod
    def forward(ctx, x, weight, bias, world_size):
        """
        前向传播：
        - x: [batch_size, in_features_per_gpu] 输入切分
        - weight: [out_features, in_features_per_gpu] 本地权重切分
        - bias: [out_features] 完整偏置（只在 rank 0 添加）
        - 输出: [batch_size, out_features] 完整输出
        """
        ctx.save_for_backward(x, weight)
        ctx.world_size = world_size

        # 本地矩阵乘法
        output = x @ weight.t()

        # all-reduce 汇总所有 GPU 的结果
        if world_size > 1:
            all_reduce(output)

        # all-reduce 后每个 GPU 都有相同的完整输出
        # 直接添加完整的偏置（每个 GPU 都加相同的值，结果正确）
        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：
        - grad_output: [batch_size, out_features] 完整输出梯度
        - 输入梯度需要在 input 维度上 all-gather
        """
        x, weight = ctx.saved_tensors
        world_size = ctx.world_size

        # 计算本地梯度
        grad_x = grad_output @ weight
        grad_weight = grad_output.t() @ x
        grad_bias = grad_output.sum(dim=0)  # 偏置梯度

        # grad_x 是切分的，但行并行的输入本来就是切分的
        # 所以这里直接返回切分的梯度即可

        return grad_x, grad_weight, grad_bias, None


class RowParallelLinear(nn.Module):
    """
    行并行线性层
    将权重按行切分，每个 GPU 持有输入维度的一部分

    特点：
    - 输入是切分的（来自列并行层或手动切分）
    - 输出是完整的（通过 all-reduce 合并）
    """

    def __init__(self, in_features, out_features, bias=True, input_is_parallel=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel

        world_size = get_world_size()
        self.world_size = world_size

        # 每个进程持有 in_features // world_size 的输入维度
        assert in_features % world_size == 0, \
            f"in_features ({in_features}) must be divisible by world_size ({world_size})"

        self.in_features_per_gpu = in_features // world_size

        # 本地权重切分
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_gpu)
        )

        if bias:
            # 偏置是完整的，会在前向传播中处理
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features  # 注意：fan_in 是完整的输入维度
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        前向传播
        x: [batch_size, in_features_per_gpu] 输入切分
        输出: [batch_size, out_features] 完整输出
        """
        output = RowParallelLinearFunction.apply(
            x, self.weight, self.bias, self.world_size
        )
        return output


# ============== TP MLP 层 ==============

class ColumnParallelLinearWithAllGather(nn.Module):
    """
    列并行线性层 + All-Gather
    输出完整的激活值

    用于需要完整输出的场景
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = ColumnParallelLinear(in_features, out_features, bias=bias)

    def forward(self, x):
        # 列并行计算，得到切分输出
        output_parallel = self.linear(x)
        # all-gather 得到完整输出
        output = all_gather(output_parallel, dim=-1)
        return output


class RowParallelLinearWithScatter(nn.Module):
    """
    Scatter + 行并行线性层
    输入是完整的，内部先切分再计算

    用于需要完整输入的场景
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = RowParallelLinear(
            in_features, out_features, bias=bias,
            input_is_parallel=False
        )

    def forward(self, x):
        # 切分输入
        world_size = get_world_size()
        if world_size > 1:
            # 在最后一维切分
            x_chunks = torch.chunk(x, world_size, dim=-1)
            x_parallel = x_chunks[get_rank()].contiguous()
        else:
            x_parallel = x

        return self.linear(x_parallel)


class TPMLP(nn.Module):
    """
    张量并行 MLP 层
    先行并行，再列并行

    结构：
    - 行并行：完整输入 -> 切分输入 -> all-reduce -> 完整输出
    - 列并行：完整输入 -> 切分输出

    注意：这种组合需要额外的通信，不如 列并行->行并行 高效
    但为了学习目的，这里展示两种组合方式
    """

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        # 行并行第一层：输入需要先 scatter，输出通过 all-reduce 得到完整结果
        self.fc1 = RowParallelLinearWithScatter(in_features, hidden_features, bias=True)

        # 列并行第二层：输入完整，输出切分
        self.fc2 = ColumnParallelLinearWithAllGather(hidden_features, out_features, bias=True)

    def forward(self, x):
        """
        前向传播
        x: [batch_size, in_features] 完整输入
        输出: [batch_size, out_features] 完整输出
        """
        # 行并行：scatter 输入，all-reduce 输出
        h = self.fc1(x)
        h = F.gelu(h)  # 激活函数

        # 列并行：完整输入，all-gather 得到完整输出
        out = self.fc2(h)

        return out


class TPMLPColumnRow(nn.Module):
    """
    张量并行 MLP 层（高效版本）
    先列并行，再行并行

    这是最常用的 TP 组合，因为：
    - 列并行输出切分，直接作为行并行输入切分
    - 中间不需要通信
    - 只在最后做一次 all-reduce

    通信对比：
    - 列并行->行并行：1 次 all-reduce
    - 行并行->列并行：1 次 all-reduce + 1 次 all-gather + 1 次 scatter
    """

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        # 列并行第一层：输入完整，输出切分
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, bias=True)

        # 行并行第二层：输入切分，输出完整
        self.fc2 = RowParallelLinear(hidden_features, out_features, bias=True)

    def forward(self, x):
        """
        前向传播
        x: [batch_size, in_features] 完整输入
        输出: [batch_size, out_features] 完整输出
        """
        # 列并行：得到切分的隐藏层
        h = self.fc1(x)
        h = F.gelu(h)  # 激活函数

        # 行并行：输入切分，输出完整（内部 all-reduce）
        out = self.fc2(h)

        return out


# ============== 验证和测试 ==============

def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' not in os.environ:
        return 0, 1

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # 设置设备
    torch.cuda.set_device(rank)

    # 初始化进程组
    dist.init_process_group(backend='nccl')

    return rank, world_size


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def verify_column_parallel(rank, world_size):
    """验证列并行线性层的正确性"""
    print(f"\n[Rank {rank}] 验证列并行线性层...")

    in_features = 64
    out_features = 128
    batch_size = 16

    # 设置相同的随机种子（包括 CUDA）
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # 创建完整的线性层（仅在 rank 0 用于比较）
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    full_linear = nn.Linear(in_features, out_features).cuda()

    # 创建列并行线性层
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    tp_linear = ColumnParallelLinear(in_features, out_features).cuda()

    # 手动切分完整权重并加载到 TP 层
    with torch.no_grad():
        weight_chunks = torch.chunk(full_linear.weight.data, world_size, dim=0)
        bias_chunks = torch.chunk(full_linear.bias.data, world_size, dim=0)
        tp_linear.weight.data.copy_(weight_chunks[rank])
        tp_linear.bias.data.copy_(bias_chunks[rank])

    # 测试输入 - 使用相同的种子确保所有 GPU 生成相同的输入
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    x = torch.randn(batch_size, in_features).cuda()

    # 完整层的前向传播
    full_output = full_linear(x)

    # TP 层的前向传播
    tp_output_parallel = tp_linear(x)
    # all-gather 得到完整输出
    tp_output = all_gather(tp_output_parallel, dim=-1)

    # 同步所有进程
    dist.barrier()

    # 比较输出 - 在所有 rank 上计算差异
    max_diff = (full_output - tp_output).abs().max().item()

    if rank == 0:
        print(f"  列并行输出最大差异: {max_diff:.6e}")
        if max_diff < 1e-5:
            print("  [PASS] 列并行前向传播正确!")
        else:
            print("  [FAIL] 列并行前向传播存在差异!")

    # 验证反向传播
    full_output.sum().backward()
    tp_output.sum().backward()

    # 收集所有 GPU 的权重梯度
    gathered_grad = all_gather(tp_linear.weight.grad.data, dim=0)

    if rank == 0:
        grad_diff = (full_linear.weight.grad - gathered_grad).abs().max().item()
        print(f"  列并行权重梯度最大差异: {grad_diff:.6e}")
        if grad_diff < 1e-5:
            print("  [PASS] 列并行反向传播正确!")
        else:
            print("  [FAIL] 列并行反向传播存在差异!")


def verify_row_parallel(rank, world_size):
    """验证行并行线性层的正确性"""
    print(f"\n[Rank {rank}] 验证行并行线性层...")

    in_features = 128
    out_features = 64
    batch_size = 16

    # 设置相同的随机种子（包括 CUDA）
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # 创建完整的线性层（仅在 rank 0 用于比较）
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    full_linear = nn.Linear(in_features, out_features).cuda()

    # 创建行并行线性层
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    tp_linear = RowParallelLinear(in_features, out_features).cuda()

    # 手动切分完整权重并加载到 TP 层
    with torch.no_grad():
        weight_chunks = torch.chunk(full_linear.weight.data, world_size, dim=1)
        tp_linear.weight.data.copy_(weight_chunks[rank])
        tp_linear.bias.data.copy_(full_linear.bias.data)

    # 测试输入 - 先生成完整输入，然后切分
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    full_x = torch.randn(batch_size, in_features).cuda()

    # 切分输入
    x_chunks = torch.chunk(full_x, world_size, dim=-1)
    x_parallel = x_chunks[rank].contiguous()

    # 完整层的前向传播
    full_output = full_linear(full_x)

    # TP 层的前向传播
    tp_output = tp_linear(x_parallel)

    # 同步所有进程
    dist.barrier()

    # 比较输出 - 在所有 rank 上计算差异
    max_diff = (full_output - tp_output).abs().max().item()

    if rank == 0:
        print(f"  行并行输出最大差异: {max_diff:.6e}")
        if max_diff < 1e-5:
            print("  [PASS] 行并行前向传播正确!")
        else:
            print("  [FAIL] 行并行前向传播存在差异!")

    # 验证反向传播
    full_output.sum().backward()
    tp_output.sum().backward()

    # 收集所有 GPU 的权重梯度
    gathered_grad = all_gather(tp_linear.weight.grad.data, dim=1)

    if rank == 0:
        grad_diff = (full_linear.weight.grad - gathered_grad).abs().max().item()
        print(f"  行并行权重梯度最大差异: {grad_diff:.6e}")
        if grad_diff < 1e-5:
            print("  [PASS] 行并行反向传播正确!")
        else:
            print("  [FAIL] 行并行反向传播存在差异!")


def verify_tp_mlp(rank, world_size):
    """验证 TP MLP 的正确性（先行并行，再列并行）"""
    print(f"\n[Rank {rank}] 验证 TP MLP (Row -> Column)...")

    in_features = 64
    hidden_features = 128
    out_features = 32
    batch_size = 16

    # 创建完整的 MLP（仅在 rank 0 用于比较）
    torch.manual_seed(42)
    class FullMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

        def forward(self, x):
            h = self.fc1(x)
            h = F.gelu(h)
            return self.fc2(h)

    full_mlp = FullMLP().cuda()

    # 创建 TP MLP
    torch.manual_seed(42)
    tp_mlp = TPMLP(in_features, hidden_features, out_features).cuda()

    # 手动切分权重
    with torch.no_grad():
        # fc1 是行并行，按输入维度切分
        fc1_weight_chunks = torch.chunk(full_mlp.fc1.weight.data, world_size, dim=1)
        tp_mlp.fc1.linear.weight.data.copy_(fc1_weight_chunks[rank])
        tp_mlp.fc1.linear.bias.data.copy_(full_mlp.fc1.bias.data)

        # fc2 是列并行，按输出维度切分
        fc2_weight_chunks = torch.chunk(full_mlp.fc2.weight.data, world_size, dim=0)
        fc2_bias_chunks = torch.chunk(full_mlp.fc2.bias.data, world_size, dim=0)
        tp_mlp.fc2.linear.weight.data.copy_(fc2_weight_chunks[rank])
        tp_mlp.fc2.linear.bias.data.copy_(fc2_bias_chunks[rank])

    # 测试输入
    torch.manual_seed(123)
    x = torch.randn(batch_size, in_features).cuda()

    # 完整 MLP 的前向传播
    full_output = full_mlp(x)

    # TP MLP 的前向传播
    tp_output = tp_mlp(x)

    # 比较输出
    if rank == 0:
        max_diff = (full_output - tp_output).abs().max().item()
        print(f"  TP MLP 输出最大差异: {max_diff:.6e}")
        if max_diff < 1e-4:
            print("  [PASS] TP MLP 前向传播正确!")
        else:
            print("  [FAIL] TP MLP 前向传播存在差异!")

    # 验证反向传播
    full_output.sum().backward()
    tp_output.sum().backward()

    gathered_fc1_weight_grad = all_gather(tp_mlp.fc1.linear.weight.grad.data, dim=1)

    # 收集 fc2 梯度（列并行，按输出维度切分）
    gathered_fc2_weight_grad = all_gather(tp_mlp.fc2.linear.weight.grad.data, dim=0)

    if rank == 0:
        fc1_weight_grad_diff = (full_mlp.fc1.weight.grad - gathered_fc1_weight_grad).abs().max().item()
        fc2_weight_grad_diff = (full_mlp.fc2.weight.grad - gathered_fc2_weight_grad).abs().max().item()

        print(f"  fc1 (行并行) 权重梯度最大差异: {fc1_weight_grad_diff:.6e}")
        print(f"  fc2 (列并行) 权重梯度最大差异: {fc2_weight_grad_diff:.6e}")

        if fc1_weight_grad_diff < 1e-4 and fc2_weight_grad_diff < 1e-4:
            print("  [PASS] TP MLP 反向传播正确!")
        else:
            print("  [FAIL] TP MLP 反向传播存在差异!")


def train_tp_mlp(rank, world_size, args):
    """使用 TP MLP 进行训练"""
    print(f"\n[Rank {rank}] 开始训练 TP MLP...")

    in_features = 64
    hidden_features = 128
    out_features = 10
    batch_size = 32
    num_epochs = args.epochs
    lr = args.lr

    # 创建模型
    torch.manual_seed(42)
    model = TPMLP(in_features, hidden_features, out_features).cuda()

    # 同步初始权重
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 创建数据集
    num_samples = 1000
    torch.manual_seed(123)
    X = torch.randn(num_samples, in_features).cuda()
    y = torch.randint(0, out_features, (num_samples,)).cuda()

    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            batch_x = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            optimizer.zero_grad()

            # 前向传播
            output = model(batch_x)

            # 计算损失（每个 GPU 独立计算）
            loss = F.cross_entropy(output, batch_y)
            total_loss += loss.item()
            num_batches += 1

            # 反向传播
            loss.backward()

            # 梯度同步（TP 中部分梯度需要 all-reduce）
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad.div_(world_size)

            optimizer.step()

        if rank == 0:
            avg_loss = total_loss / num_batches
            print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    if rank == 0:
        print("  训练完成!")


def main():
    parser = argparse.ArgumentParser(description='Tensor Parallelism Demo')
    parser.add_argument('--mode', type=str, default='verify',
                        choices=['verify', 'train'],
                        help='Running mode')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    args = parser.parse_args()

    # 初始化分布式
    rank, world_size = setup_distributed()

    if rank == 0:
        print("=" * 60)
        print("张量并行 (Tensor Parallelism) 演示")
        print(f"World Size: {world_size}")
        print("=" * 60)

    if args.mode == 'verify':
        # 验证各层正确性
        verify_column_parallel(rank, world_size)
        verify_row_parallel(rank, world_size)
        verify_tp_mlp(rank, world_size)

        if rank == 0:
            print("\n" + "=" * 60)
            print("所有验证完成!")
            print("=" * 60)

    elif args.mode == 'train':
        # 训练演示
        train_tp_mlp(rank, world_size, args)

    # 清理
    cleanup_distributed()


if __name__ == "__main__":
    main()