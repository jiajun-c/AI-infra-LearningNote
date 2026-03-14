"""
数据并行 (Data Parallelism) 实现
使用 torch.autograd.Function 手写前向传播、反向传播
使用 NCCL (torch.distributed) 实现多 GPU 梯度同步

核心思想：
1. 每个进程管理一个 GPU
2. 前向传播：各进程独立计算
3. 反向传播：各进程独立计算梯度，然后使用 NCCL all-reduce 同步
4. 参数更新：所有进程使用同步后的梯度更新参数

运行方式：
  # 训练模式
  torchrun --nproc_per_node=2 dp.py --mode train

  # 验证正确性（推荐）
  torchrun --nproc_per_node=2 dp.py --mode verify

  # 单卡训练
  python dp.py --mode train_single

  # 性能测试
  python dp.py --mode benchmark
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import time
import os
import argparse


# ============== 手写算子实现 ==============

class LinearFunction(torch.autograd.Function):
    """
    手写线性层的前向和反向传播
    y = x @ W.t() + b
    """
    @staticmethod
    def forward(ctx, x, weight, bias):
        # 使用 save_for_backward 保存反向传播需要的张量
        ctx.save_for_backward(x, weight, bias)
        output = x @ weight.t() + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors

        grad_x = grad_output @ weight
        grad_weight = grad_output.t() @ x
        grad_bias = grad_output.sum(dim=0)
        # print(f"grad_out {grad_output.shape}")
        # print(f"x {x.shape}")
        # print(f"grad_weight {grad_weight.shape}")

        return grad_x, grad_weight, grad_bias


class ReLUFunction(torch.autograd.Function):
    """手写 ReLU 的前向和反向传播"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.maximum(x, torch.zeros_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x <= 0] = 0
        return grad_input


class CrossEntropyLossFunction(torch.autograd.Function):
    """手写 CrossEntropyLoss 的前向和反向传播"""
    @staticmethod
    def forward(ctx, logits, targets):
        batch_size = logits.shape[0]

        # 数值稳定的 log softmax
        x_max = logits.max(dim=-1, keepdim=True)[0]
        exp_logits = torch.exp(logits - x_max)
        sum_exp = exp_logits.sum(dim=-1, keepdim=True)
        log_probs = logits - x_max - torch.log(sum_exp)

        loss = -log_probs[torch.arange(batch_size, device=logits.device), targets].mean()

        ctx.save_for_backward(log_probs, targets)
        ctx.batch_size = batch_size

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        log_probs, targets = ctx.saved_tensors
        batch_size = ctx.batch_size

        probs = torch.exp(log_probs)
        grad_logits = probs.clone()
        grad_logits[torch.arange(batch_size, device=probs.device), targets] -= 1
        grad_logits = grad_logits / batch_size * grad_output

        return grad_logits, None


class HandwrittenLinear(nn.Module):
    """使用手写 Function 的线性层"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (2.0 / in_features) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return LinearFunction.apply(x, self.weight, self.bias)


class HandwrittenReLU(nn.Module):
    """使用手写 Function 的 ReLU"""
    def forward(self, x):
        return ReLUFunction.apply(x)


class HandwrittenCrossEntropyLoss(nn.Module):
    """使用手写 Function 的 CrossEntropyLoss"""
    def forward(self, logits, targets):
        return CrossEntropyLossFunction.apply(logits, targets)


class SimpleMLP(nn.Module):
    """使用手写算子的 MLP 模型"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = HandwrittenLinear(input_dim, hidden_dim)
        self.relu1 = HandwrittenReLU()
        self.linear2 = HandwrittenLinear(hidden_dim, hidden_dim)
        self.relu2 = HandwrittenReLU()
        self.linear3 = HandwrittenLinear(hidden_dim, output_dim)
        self.loss_fn = HandwrittenCrossEntropyLoss()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

    def compute_loss(self, logits, targets):
        return self.loss_fn(logits, targets)


# ============== NCCL 梯度同步函数 ==============

def all_reduce_gradients(model):
    """
    使用 NCCL all-reduce 同步所有进程的梯度
    对每个参数的梯度进行 all-reduce 求和，然后除以进程数得到平均值
    """
    world_size = dist.get_world_size()

    for param in model.parameters():
        if param.grad is not None:
            # 使用 all-reduce 对梯度求和
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            # 除以进程数得到平均梯度
            param.grad.div_(world_size)


def broadcast_parameters(model, src_rank=0):
    """
    从源进程广播参数到所有进程
    确保所有进程使用相同的初始参数
    """
    for param in model.parameters():
        dist.broadcast(param.data, src=src_rank)


# ============== 分布式训练函数 ==============

def setup_distributed(rank, world_size):
    """
    初始化分布式环境
    优先使用 NCCL 后端，失败时回退到 gloo
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # 设置当前设备
    torch.cuda.set_device(rank)

    # 尝试使用 NCCL，失败则回退到 gloo
    backend = 'nccl'
    try:
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
        print(f"[Rank {rank}] Initialized with NCCL backend")
    except Exception as e:
        print(f"[Rank {rank}] NCCL failed ({e}), falling back to gloo")
        backend = 'gloo'
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
        print(f"[Rank {rank}] Initialized with gloo backend")

    return backend


def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()


def train_worker(rank, world_size, args):
    """
    每个进程的训练函数
    使用 NCCL 进行梯度同步
    """
    # 检查 GPU 是否可用
    if not torch.cuda.is_available():
        print(f"[Rank {rank}] CUDA not available!")
        return

    # 初始化分布式环境
    backend = setup_distributed(rank, world_size)

    device = torch.device(f'cuda:{rank}')
    torch.manual_seed(42)  # 确保所有进程初始化相同的参数

    # 模型参数
    input_dim = 784
    hidden_dim = 256
    output_dim = 10
    lr = args.lr * world_size
    batch_size = args.batch_size
    num_epochs = args.epochs

    # 创建模型
    model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)

    # 从 rank 0 广播参数到所有进程
    broadcast_parameters(model, src_rank=0)

    # 创建优化器
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 创建数据集
    num_samples = 10000
    torch.manual_seed(42)
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))
    dataset = TensorDataset(X, y)

    # 使用 DistributedSampler 确保数据分布到各进程
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        sampler.set_epoch(epoch)  # 确保每个 epoch 数据随机

        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # print(f"batch_x {batch_x.shape}")
            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            logits = model(batch_x)

            # 计算损失
            loss = model.compute_loss(logits, batch_y)
            total_loss += loss.item()

            # 反向传播
            loss.backward()

            # 使用 NCCL 同步梯度
            all_reduce_gradients(model)

            # 更新参数
            optimizer.step()

            # 计算准确率
            predictions = logits.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    if rank == 0:
        print("\nDistributed Training with NCCL completed!")

    cleanup_distributed()


def train_single_gpu(args):
    """
    单 GPU 训练（作为对比）
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("=" * 60)
    print("Single GPU Training (Baseline)")
    print("=" * 60)

    input_dim = 784
    hidden_dim = 256
    output_dim = 10
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.epochs

    torch.manual_seed(42)
    model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 创建数据集
    num_samples = 10000
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = model.compute_loss(logits, batch_y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    print("\nSingle GPU Training completed!")


def verify_worker(rank, world_size, args):
    """
    分布式验证函数：在每个进程中运行，验证多卡训练的正确性
    """
    if not torch.cuda.is_available():
        print(f"[Rank {rank}] CUDA not available!")
        return

    backend = setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # 模型参数
    input_dim = 784
    hidden_dim = 128
    output_dim = 10
    num_steps = 5
    batch_size = 64
    lr = 0.01

    # 所有进程使用相同的种子初始化相同的模型
    torch.manual_seed(42)
    model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 所有进程使用相同的数据
    torch.manual_seed(123)
    X = torch.randn(batch_size, input_dim).to(device)
    y = torch.randint(0, output_dim, (batch_size,)).to(device)

    # 记录每步的 loss
    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()
        logits = model(X)
        loss = model.compute_loss(logits, y)
        losses.append(loss.item())
        loss.backward()

        # 使用 NCCL 同步梯度
        all_reduce_gradients(model)

        optimizer.step()

    # 同步所有进程，确保训练完成
    dist.barrier()

    # 收集所有进程的最终 loss 进行比较
    losses_tensor = torch.tensor(losses, device=device)
    losses_list = [torch.zeros_like(losses_tensor) for _ in range(world_size)]
    dist.all_gather(losses_list, losses_tensor)

    # 收集所有进程的参数进行比较
    param_dict = {name: param.data.clone() for name, param in model.named_parameters()}

    if rank == 0:
        print("\n" + "=" * 70)
        print("Verification Results:")
        print("=" * 70)

        # 检查所有进程的 loss 是否一致
        all_losses_match = True
        for i in range(1, world_size):
            if not torch.allclose(losses_list[0], losses_list[i], atol=1e-6):
                all_losses_match = False
                print(f"[WARNING] Loss mismatch between rank 0 and rank {i}!")
                print(f"  Rank 0 losses: {losses_list[0].tolist()}")
                print(f"  Rank {i} losses: {losses_list[i].tolist()}")

        if all_losses_match:
            print(f"[PASS] All ranks have consistent losses!")
            print(f"  Final loss: {losses[-1]:.6f}")

        print()

    # 收集所有进程的参数到 rank 0 进行比较
    for name, param in model.named_parameters():
        gathered_params = [torch.zeros_like(param.data) for _ in range(world_size)]
        dist.all_gather(gathered_params, param.data)

        if rank == 0:
            params_match = True
            for i in range(1, world_size):
                if not torch.allclose(gathered_params[0], gathered_params[i], atol=1e-6):
                    params_match = False
                    print(f"[WARNING] Parameter '{name}' mismatch between rank 0 and rank {i}!")
                    break

            if params_match:
                print(f"[PASS] Parameter '{name}' is consistent across all ranks")

    dist.barrier()

    # 在 rank 0 上运行单 GPU 基准测试并比较
    if rank == 0:
        print("\n[3] Comparing with Single GPU baseline...")

        torch.manual_seed(42)
        model_single = SimpleMLP(input_dim, hidden_dim, output_dim).to('cuda:0')
        optimizer_single = optim.SGD(model_single.parameters(), lr=lr)

        torch.manual_seed(123)
        X_single = torch.randn(batch_size, input_dim).to('cuda:0')
        y_single = torch.randint(0, output_dim, (batch_size,)).to('cuda:0')

        single_losses = []
        for _ in range(num_steps):
            optimizer_single.zero_grad()
            logits = model_single(X_single)
            loss = model_single.compute_loss(logits, y_single)
            single_losses.append(loss.item())
            loss.backward()
            optimizer_single.step()

        print(f"   Single GPU final loss: {single_losses[-1]:.6f}")
        print(f"   Distributed final loss: {losses[-1]:.6f}")

        # 比较单 GPU 和分布式的 loss 差异
        loss_diff = abs(single_losses[-1] - losses[-1])
        if loss_diff < 1e-5:
            print(f"[PASS] Single GPU and Distributed losses match! (diff: {loss_diff:.2e})")
        else:
            print(f"[WARNING] Loss difference: {loss_diff:.6f}")

        # 比较参数
        print("\n[4] Comparing model parameters...")
        all_params_match = True
        for (name_d, param_d), (name_s, param_s) in zip(model.named_parameters(), model_single.named_parameters()):
            if not torch.allclose(param_d.cpu(), param_s.cpu(), atol=1e-5):
                all_params_match = False
                print(f"[WARNING] Parameter '{name_d}' differs between distributed and single GPU!")
                print(f"  Max diff: {(param_d.cpu() - param_s.cpu()).abs().max().item():.6e}")

        if all_params_match:
            print("[PASS] All parameters match between distributed and single GPU!")

        print("\n" + "=" * 70)
        print("Verification Complete!")
        print("=" * 70)

    cleanup_distributed()


def verify_nccl_correctness(args):
    """
    验证 NCCL 数据并行的正确性
    对比单 GPU 和分布式训练的结果
    使用方法: torchrun --nproc_per_node=2 dp.py --mode verify
    """
    # 检查是否在分布式环境中运行
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        if world_size >= 2:
            verify_worker(rank, world_size, args)
            return

    # 非分布式环境下的提示
    print("=" * 70)
    print("NCCL Data Parallel Correctness Verification")
    print("=" * 70)

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Need at least 2 GPUs for verification, skipping...")
        return

    print(f"Detected {world_size} GPUs")
    print("\nTo verify correctness, run with torchrun:")
    print(f"  torchrun --nproc_per_node={world_size} dp.py --mode verify")
    print("\n" + "=" * 70)


def benchmark_nccl_performance(args):
    """
    性能测试：对比单 GPU 和 NCCL 数据并行的性能
    """
    print("=" * 70)
    print("NCCL Performance Benchmark")
    print("=" * 70)

    world_size = torch.cuda.device_count()
    print(f"Available GPUs: {world_size}")

    if world_size < 2:
        print("Need at least 2 GPUs for comparison, running single GPU benchmark only...")
        train_single_gpu(args)
        return

    # 单 GPU 性能
    print("\n[1] Single GPU Performance")
    device = 'cuda:0'
    input_dim = 784
    hidden_dim = 512
    output_dim = 10
    batch_size = 256
    num_iterations = 50
    lr = 0.01

    torch.manual_seed(42)
    model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    X = torch.randn(batch_size, input_dim).to(device)
    y = torch.randint(0, output_dim, (batch_size,)).to(device)

    # 预热
    for _ in range(10):
        optimizer.zero_grad()
        logits = model(X)
        loss = model.compute_loss(logits, y)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_iterations):
        optimizer.zero_grad()
        logits = model(X)
        loss = model.compute_loss(logits, y)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    single_time = time.time() - start_time

    print(f"   Time: {single_time:.4f}s ({single_time/num_iterations*1000:.2f}ms/iter)")

    print("\n[2] For NCCL distributed performance, run:")
    print(f"   torchrun --nproc_per_node={world_size} dp.py --mode train")

    print("\n" + "=" * 70)


def run_with_spawn(args):
    """
    使用 mp.spawn 启动多进程训练
    """
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Need at least 2 GPUs for distributed training")
        print("Running single GPU training instead...")
        train_single_gpu(args)
        return

    print(f"Starting distributed training with {world_size} GPUs using NCCL")
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


def main():
    parser = argparse.ArgumentParser(description='Data Parallel Training with NCCL')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'train_single', 'verify', 'benchmark', 'spawn'],
                        help='Running mode')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    args = parser.parse_args()

    if args.mode == 'train':
        # 使用 torchrun 启动时运行此模式
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))

        if world_size > 1:
            train_worker(rank, world_size, args)
        else:
            print("World size is 1, running single GPU training...")
            train_single_gpu(args)

    elif args.mode == 'train_single':
        train_single_gpu(args)

    elif args.mode == 'verify':
        verify_nccl_correctness(args)

    elif args.mode == 'benchmark':
        benchmark_nccl_performance(args)

    elif args.mode == 'spawn':
        run_with_spawn(args)


if __name__ == "__main__":
    main()