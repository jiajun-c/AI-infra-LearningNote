"""
流水线并行 (Pipeline Parallelism) 实现
实现 GPipe 风格的流水线并行

核心思想：
1. 将模型按层切分到不同设备（Stage）
2. 将 mini-batch 切分为多个 micro-batch
3. 前向传播：按顺序在各 stage 执行
4. 反向传播：按逆序在各 stage 执行
5. 流水线调度：让不同 micro-batch 在不同 stage 上并行执行

运行方式：
  torchrun --nproc_per_node=2 pp.py --mode verify
  torchrun --nproc_per_node=2 pp.py --mode train

注意：PP 需要多 GPU 环境，每个 rank 对应一个 stage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
import argparse
import math
from typing import List, Tuple, Optional


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


def send(tensor: torch.Tensor, dst: int):
    """发送张量到目标进程"""
    if not dist.is_initialized():
        return
    dist.send(tensor.contiguous(), dst=dst)


def recv(tensor: torch.Tensor, src: int):
    """从源进程接收张量"""
    if not dist.is_initialized():
        return
    dist.recv(tensor, src=src)


def isend(tensor: torch.Tensor, dst: int):
    """异步发送"""
    if not dist.is_initialized():
        return None
    return dist.isend(tensor.contiguous(), dst=dst)


def irecv(tensor: torch.Tensor, src: int):
    """异步接收"""
    if not dist.is_initialized():
        return None
    return dist.irecv(tensor, src=src)


# ============== 模型分割 ==============

class PipelineStage(nn.Module):
    """
    流水线的一个阶段（Stage）
    每个 stage 包含模型的一部分层

    在 PP 中：
    - Stage 0: 接收原始输入
    - Stage i (0 < i < n-1): 接收前一 stage 的输出
    - Stage n-1: 产生最终输出
    """

    def __init__(self, layers: nn.ModuleList, stage_id: int, num_stages: int):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.num_stages = num_stages

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def is_first_stage(self) -> bool:
        return self.stage_id == 0

    def is_last_stage(self) -> bool:
        return self.stage_id == self.num_stages - 1


def split_model_into_stages(model: nn.Module, num_stages: int) -> List[nn.ModuleList]:
    """
    将模型分割成多个 stage

    Args:
        model: 原始模型
        num_stages: stage 数量

    Returns:
        每个 stage 的层列表
    """
    # 收集所有层
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.ReLU, nn.GELU, nn.Dropout, nn.LayerNorm)):
            layers.append((name, module))

    # 如果没有找到层，尝试直接使用 children
    if not layers:
        layers = [(f'layer_{i}', m) for i, m in enumerate(model.children())]

    # 分割层到各 stage
    layers_per_stage = len(layers) // num_stages
    stages = []

    for i in range(num_stages):
        start_idx = i * layers_per_stage
        if i == num_stages - 1:
            # 最后一个 stage 获取剩余所有层
            stage_layers = layers[start_idx:]
        else:
            stage_layers = layers[start_idx:start_idx + layers_per_stage]

        # 提取模块
        modules = nn.ModuleList([m for _, m in stage_layers])
        stages.append(modules)

    return stages


# ============== Micro-batch 处理 ==============

class MicroBatch:
    """
    Micro-batch 数据结构
    保存一个 micro-batch 的输入、输出和梯度
    """

    def __init__(self, micro_id: int):
        self.micro_id = micro_id
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.grad_output: Optional[torch.Tensor] = None

    def save_for_backward(self, tensor: torch.Tensor):
        """保存需要梯度的张量"""
        self.input = tensor.detach().clone()
        self.input.requires_grad_(True)


def split_batch(x: torch.Tensor, num_micro_batches: int) -> List[torch.Tensor]:
    """
    将 mini-batch 分割成多个 micro-batch

    Args:
        x: [batch_size, ...] 输入张量
        num_micro_batches: micro-batch 数量

    Returns:
        micro-batch 列表
    """
    batch_size = x.shape[0]
    assert batch_size % num_micro_batches == 0, \
        f"Batch size ({batch_size}) must be divisible by num_micro_batches ({num_micro_batches})"

    return torch.chunk(x, num_micro_batches, dim=0)


# ============== GPipe 流水线调度 ==============

class GPipeScheduler:
    """
    GPipe 流水线调度器

    GPipe 调度策略：
    1. 前向阶段：依次处理所有 micro-batch 的前向传播
    2. 反向阶段：依次处理所有 micro-batch 的反向传播

    示例（4个 stage，4个 micro-batch）：
    时间 ->

    前向阶段:
    Stage 0: [F0] [F1] [F2] [F3]
    Stage 1:      [F0] [F1] [F2] [F3]
    Stage 2:           [F0] [F1] [F2] [F3]
    Stage 3:                [F0] [F1] [F2] [F3]

    反向阶段:
    Stage 3:                               [B0] [B1] [B2] [B3]
    Stage 2:                          [B0] [B1] [B2] [B3]
    Stage 1:                     [B0] [B1] [B2] [B3]
    Stage 0:                [B0] [B1] [B2] [B3]

    其中 F_i 表示第 i 个 micro-batch 的前向传播
         B_i 表示第 i 个 micro-batch 的反向传播
    """

    def __init__(self, num_micro_batches: int, num_stages: int):
        self.num_micro_batches = num_micro_batches
        self.num_stages = num_stages

    def get_forward_schedule(self) -> List[Tuple[int, int]]:
        """
        获取前向传播调度
        Returns: List of (stage_id, micro_batch_id)
        """
        schedule = []
        for micro_id in range(self.num_micro_batches):
            for stage_id in range(self.num_stages):
                schedule.append((stage_id, micro_id))
        return schedule

    def get_backward_schedule(self) -> List[Tuple[int, int]]:
        """
        获取反向传播调度
        Returns: List of (stage_id, micro_batch_id)
        """
        schedule = []
        for micro_id in range(self.num_micro_batches):
            for stage_id in range(self.num_stages - 1, -1, -1):
                schedule.append((stage_id, micro_id))
        return schedule


# ============== 流水线并行模型 ==============

class PipelineParallel(nn.Module):
    """
    流水线并行模型

    将模型分割到多个 GPU 上，使用 GPipe 风格的调度

    使用方式：
    1. 创建完整模型
    2. 用 PipelineParallel 包装
    3. 正常调用 forward 和 backward
    """

    def __init__(self, model: nn.Module, num_micro_batches: int = 4):
        super().__init__()
        self.num_micro_batches = num_micro_batches

        # 获取分布式信息
        self.world_size = get_world_size()
        self.rank = get_rank()

        # 分割模型
        if self.world_size > 1:
            stages = split_model_into_stages(model, self.world_size)
            self.stage = PipelineStage(
                stages[self.rank],
                self.rank,
                self.world_size
            )
        else:
            # 单 GPU 模式：整个模型作为一个 stage
            layers = nn.ModuleList([m for m in model.children()])
            self.stage = PipelineStage(layers, 0, 1)

        # 调度器
        self.scheduler = GPipeScheduler(num_micro_batches, max(self.world_size, 1))

        # 缓存 micro-batch 数据
        self.micro_batches: List[MicroBatch] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（包含流水线调度）

        Args:
            x: [batch_size, ...] 输入张量

        Returns:
            输出张量（仅在最后一个 stage 有效）
        """
        # 分割输入
        if self.stage.is_first_stage():
            micro_inputs = split_batch(x, self.num_micro_batches)
        else:
            micro_inputs = [None] * self.num_micro_batches

        # 初始化 micro-batch 存储
        self.micro_batches = [MicroBatch(i) for i in range(self.num_micro_batches)]

        # 存储输出
        outputs = []

        # ===== 前向传播阶段 =====
        for micro_id in range(self.num_micro_batches):
            micro_batch = self.micro_batches[micro_id]

            # 接收输入（非第一个 stage）
            if not self.stage.is_first_stage():
                recv_input = torch.empty_like(
                    micro_inputs[0] if micro_inputs[0] is not None else torch.randn(1)
                )
                # 这里简化处理，实际需要知道输入 shape
                # 在实际实现中，通常会在第一次通信时传递 shape 信息

            # 获取当前 micro-batch 输入
            if self.stage.is_first_stage():
                micro_input = micro_inputs[micro_id].clone()
                micro_input.requires_grad_(True)
            else:
                # 接收前一 stage 的输出
                micro_input = self._recv_from_prev_stage(micro_inputs[0])

            # 保存输入用于反向传播
            micro_batch.input = micro_input

            # 执行当前 stage 的前向传播
            micro_output = self.stage(micro_input)
            micro_batch.output = micro_output

            # 发送输出到下一 stage（非最后一个 stage）
            if not self.stage.is_last_stage():
                self._send_to_next_stage(micro_output)
            else:
                outputs.append(micro_output)

        # 返回输出（仅最后一个 stage 有意义）
        if self.stage.is_last_stage():
            return torch.cat(outputs, dim=0)
        else:
            return torch.empty(1)  # 占位符

    def _send_to_next_stage(self, tensor: torch.Tensor):
        """发送张量到下一个 stage"""
        next_rank = self.rank + 1
        send(tensor, next_rank)

    def _recv_from_prev_stage(self, ref_tensor: torch.Tensor) -> torch.Tensor:
        """从前一个 stage 接收张量"""
        prev_rank = self.rank - 1
        recv_tensor = torch.empty_like(ref_tensor)
        recv(recv_tensor, prev_rank)
        recv_tensor.requires_grad_(True)
        return recv_tensor

    def parameters(self, recurse: bool = True):
        return self.stage.parameters(recurse)


# ============== 简化版流水线实现（单进程多 GPU）==============

class SimplePipelineParallel:
    """
    简化版流水线并行实现
    使用单进程多 GPU，便于理解和调试

    这个实现更接近真实的 GPipe，展示了：
    1. 模型如何分割到不同 GPU
    2. Micro-batch 如何在不同 stage 间流动
    3. 前向和反向传播的流水线调度
    """

    def __init__(self, model: nn.Module, num_micro_batches: int, devices: List[int]):
        """
        Args:
            model: 要并行化的模型
            num_micro_batches: micro-batch 数量
            devices: GPU 设备列表，如 [0, 1, 2, 3]
        """
        self.num_micro_batches = num_micro_batches
        self.devices = devices
        self.num_stages = len(devices)

        # 分割模型
        stages = split_model_into_stages(model, self.num_stages)

        # 将每个 stage 放到对应 GPU
        self.stages = []
        for i, stage_layers in enumerate(stages):
            stage = PipelineStage(stage_layers, i, self.num_stages)
            stage = stage.to(f'cuda:{devices[i]}')
            self.stages.append(stage)

        # 存储中间激活值（用于反向传播）
        self.activations: List[List[torch.Tensor]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（GPipe 风格）

        Args:
            x: [batch_size, ...] 输入张量

        Returns:
            输出张量
        """
        # 分割输入
        micro_inputs = split_batch(x, self.num_micro_batches)

        # 初始化激活值存储
        self.activations = [[] for _ in range(self.num_stages)]

        # 存储所有 micro-batch 的输出
        outputs = []

        # 前向传播：依次处理每个 micro-batch
        for micro_id, micro_input in enumerate(micro_inputs):
            # 保存每个 stage 的激活值
            activation = micro_input.to(f'cuda:{self.devices[0]}')
            activation.requires_grad_(True)

            # 逐 stage 执行前向传播
            for stage_id, stage in enumerate(self.stages):
                # 保存激活值
                self.activations[stage_id].append(activation)

                # 将输入移动到当前 stage 的设备
                target_device = f'cuda:{self.devices[stage_id]}'
                stage_input = activation.detach().to(target_device)

                # 执行前向传播
                activation = stage(stage_input)

                if stage_id < self.num_stages - 1:
                    # 需要梯度
                    activation.requires_grad_(True)
                else:
                    # 最后一个 stage 保存输出
                    outputs.append(activation)

        return torch.cat(outputs, dim=0)

    def backward(self, grad_output: torch.Tensor):
        """
        反向传播（GPipe 风格）

        Args:
            grad_output: 输出梯度
        """
        # 分割梯度
        grad_micros = split_batch(grad_output, self.num_micro_batches)

        # 反向传播：依次处理每个 micro-batch（逆序）
        for micro_id in range(self.num_micro_batches - 1, -1, -1):
            grad = grad_micros[micro_id]

            # 逆序执行 stage 反向传播
            for stage_id in range(self.num_stages - 1, -1, -1):
                activation = self.activations[stage_id][micro_id]

                with torch.cuda.device(self.devices[stage_id]):
                    if stage_id == self.num_stages - 1:
                        # 最后一个 stage：直接反向传播
                        activation.backward(grad)
                    else:
                        # 中间 stage：需要保存梯度传给前一 stage
                        if activation.grad is not None:
                            activation.backward(activation.grad)

    def parameters(self):
        """获取所有参数"""
        params = []
        for stage in self.stages:
            params.extend(stage.parameters())
        return params


# ============== 测试模型 ==============

class MLPModel(nn.Module):
    """
    用于测试的 MLP 模型
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 4):
        super().__init__()
        layers = []

        # 第一层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # 中间层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ============== 验证函数 ==============

def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' not in os.environ:
        return 0, 1

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # 设置设备
    torch.cuda.set_device(rank)

    # 初始化进程组
    dist.init_process_group(backend='gloo')  # 使用 gloo，便于 CPU 通信

    return rank, world_size


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def verify_pipeline_correctness():
    """
    验证流水线并行的正确性
    对比 PP 实现与普通实现的输出
    """
    print("=" * 60)
    print("流水线并行正确性验证")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过验证")
        return

    # 参数
    input_dim = 64
    hidden_dim = 128
    output_dim = 10
    num_layers = 4
    batch_size = 16
    num_micro_batches = 4

    # 检查可用 GPU 数量
    num_gpus = torch.cuda.device_count()
    devices = list(range(min(num_gpus, 4)))

    if len(devices) < 2:
        print(f"需要至少 2 个 GPU，当前只有 {len(devices)} 个")
        print("使用模拟模式...")
        devices = [0, 0]  # 在同一 GPU 上模拟

    print(f"\n配置:")
    print(f"  输入维度: {input_dim}")
    print(f"  隐藏维度: {hidden_dim}")
    print(f"  输出维度: {output_dim}")
    print(f"  层数: {num_layers}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Micro-batches: {num_micro_batches}")
    print(f"  Devices: {devices}")

    # 创建模型
    torch.manual_seed(42)
    model = MLPModel(input_dim, hidden_dim, output_dim, num_layers)

    # 创建输入
    torch.manual_seed(123)
    x = torch.randn(batch_size, input_dim)

    # ===== 普通前向传播 =====
    print("\n[1] 普通前向传播...")
    model_full = model.to(f'cuda:{devices[0]}')
    x_cuda = x.to(f'cuda:{devices[0]}')

    with torch.no_grad():
        output_full = model_full(x_cuda)

    print(f"  输出 shape: {output_full.shape}")
    print(f"  输出均值: {output_full.mean().item():.4f}")

    # ===== 流水线前向传播 =====
    print("\n[2] 流水线前向传播...")

    # 重新创建模型（重新初始化）
    torch.manual_seed(42)
    model_pp = MLPModel(input_dim, hidden_dim, output_dim, num_layers)

    # 使用简化版流水线
    pp = SimplePipelineParallel(model_pp, num_micro_batches, devices)

    with torch.no_grad():
        output_pp = pp.forward(x)

    print(f"  输出 shape: {output_pp.shape}")
    print(f"  输出均值: {output_pp.mean().item():.4f}")

    # ===== 比较结果 =====
    print("\n[3] 结果比较...")

    # 将流水线输出移到第一个 GPU
    output_pp = output_pp.to(f'cuda:{devices[0]}')

    max_diff = (output_full - output_pp).abs().max().item()
    mean_diff = (output_full - output_pp).abs().mean().item()

    print(f"  最大差异: {max_diff:.6e}")
    print(f"  平均差异: {mean_diff:.6e}")

    if max_diff < 1e-5:
        print("  [PASS] 前向传播正确!")
    else:
        print("  [FAIL] 前向传播存在差异!")
        print("  注意: 差异可能来自浮点数精度问题")

    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)


def train_with_pipeline():
    """
    使用流水线并行训练模型
    """
    print("=" * 60)
    print("流水线并行训练演示")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过训练")
        return

    # 参数
    input_dim = 64
    hidden_dim = 128
    output_dim = 10
    num_layers = 4
    batch_size = 32
    num_micro_batches = 4
    num_epochs = 5
    learning_rate = 0.01

    # 检查可用 GPU
    num_gpus = torch.cuda.device_count()
    devices = list(range(min(num_gpus, 4)))

    if len(devices) < 2:
        print(f"需要至少 2 个 GPU，当前只有 {len(devices)} 个")
        print("使用模拟模式...")
        devices = [0, 0]

    print(f"\n配置:")
    print(f"  Devices: {devices}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Micro-batches: {num_micro_batches}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning Rate: {learning_rate}")

    # 创建模型和数据
    torch.manual_seed(42)
    model = MLPModel(input_dim, hidden_dim, output_dim, num_layers)

    # 创建流水线
    pp = SimplePipelineParallel(model, num_micro_batches, devices)

    # 创建数据集
    num_samples = 500
    torch.manual_seed(123)
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))

    # 优化器
    optimizer = torch.optim.SGD(pp.parameters(), lr=learning_rate)

    # 训练循环
    print("\n开始训练...")
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            batch_x = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            # 前向传播
            output = pp.forward(batch_x)

            # 计算损失（在最后一个 GPU 上）
            output = output.to(f'cuda:{devices[-1]}')
            batch_y = batch_y.to(f'cuda:{devices[-1]}')
            loss = F.cross_entropy(output, batch_y)

            total_loss += loss.item()
            num_batches += 1

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 更新参数
            optimizer.step()

        avg_loss = total_loss / num_batches
        print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("\n训练完成!")


def demonstrate_schedule():
    """
    演示流水线调度
    """
    print("=" * 60)
    print("流水线调度演示 (GPipe)")
    print("=" * 60)

    num_micro_batches = 4
    num_stages = 4

    scheduler = GPipeScheduler(num_micro_batches, num_stages)

    print(f"\n配置: {num_stages} stages, {num_micro_batches} micro-batches")

    # 前向调度
    print("\n[前向传播调度]")
    forward_schedule = scheduler.get_forward_schedule()
    print("  时间步 -> (stage_id, micro_batch_id)")
    for t, (stage_id, micro_id) in enumerate(forward_schedule):
        print(f"    t={t:2d}: Stage {stage_id}, Micro-batch {micro_id}")

    # 反向调度
    print("\n[反向传播调度]")
    backward_schedule = scheduler.get_backward_schedule()
    print("  时间步 -> (stage_id, micro_batch_id)")
    for t, (stage_id, micro_id) in enumerate(backward_schedule):
        print(f"    t={t:2d}: Stage {stage_id}, Micro-batch {micro_id}")

    # 可视化
    print("\n[流水线可视化]")
    print("前向阶段 (F_i = Micro-batch i 前向):")
    for stage_id in range(num_stages):
        row = f"  Stage {stage_id}: "
        time_offset = stage_id
        for t in range(num_micro_batches + num_stages - 1):
            micro_id = t - time_offset
            if 0 <= micro_id < num_micro_batches:
                row += f"[F{micro_id}] "
            else:
                row += "      "
        print(row)

    print("\n反向阶段 (B_i = Micro-batch i 反向):")
    for stage_id in range(num_stages):
        row = f"  Stage {stage_id}: "
        time_offset = num_stages - 1 - stage_id
        for t in range(num_micro_batches + num_stages - 1):
            micro_id = t - time_offset
            if 0 <= micro_id < num_micro_batches:
                row += f"[B{micro_id}] "
            else:
                row += "      "
        print(row)


def main():
    parser = argparse.ArgumentParser(description='Pipeline Parallelism Demo')
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['verify', 'train', 'demo', 'all'],
                        help='Running mode')
    args = parser.parse_args()

    if args.mode == 'verify':
        verify_pipeline_correctness()
    elif args.mode == 'train':
        train_with_pipeline()
    elif args.mode == 'demo':
        demonstrate_schedule()
    elif args.mode == 'all':
        demonstrate_schedule()
        print("\n")
        verify_pipeline_correctness()
        print("\n")
        train_with_pipeline()


if __name__ == "__main__":
    main()