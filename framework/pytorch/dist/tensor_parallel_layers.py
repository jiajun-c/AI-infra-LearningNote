import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """
    列并行线性层 (Column Parallel Linear Layer).
    
    通常用于 Transformer MLP 的第一层 (Gate/Up Projection) 或 Attention 的 QKV 投影。
    它接收完整的输入 (Replicated Input)，并输出被切分的张量 (Sharded Output)。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        process_group: dist.ProcessGroup = None,
    ):
        super().__init__()
        self.process_group = process_group
        self.tp_rank = dist.get_rank(group=process_group)
        self.tp_size = dist.get_world_size(group=process_group)

        # 1. 维度切分逻辑
        # Column Parallel 切分的是 输出维度 (output_size)
        assert output_size % self.tp_size == 0, f"Output size {output_size} must be divisible by TP size {self.tp_size}"
        
        self.output_size_per_partition = output_size // self.tp_size
        self.input_size = input_size

        # 2. 定义本地权重
        # PyTorch 的 nn.Linear 权重形状是 [out_features, in_features]
        # Column Parallel: [output_size // tp_size, input_size]
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        
        # 3. Bias 处理
        # Bias 也是沿着输出维度切分的，每个 Rank 只有一部分 Bias
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # Column Parallel 切分的是 out_features (dim=0)
        tp_dim = 0
        shard_size = self.output_size_per_partition
        start_idx = self.tp_rank * shard_size
        
        # 从完整权重中切出当前 Rank 负责的行
        loaded_shard = loaded_weight.narrow(tp_dim, start_idx, shard_size)
        param.data.copy_(loaded_shard)

    def bias_loader(self, param: nn.Parameter, loaded_bias: torch.Tensor):
        # Bias 也要切分
        tp_dim = 0
        shard_size = self.output_size_per_partition
        start_idx = self.tp_rank * shard_size
        loaded_shard = loaded_bias.narrow(tp_dim, start_idx, shard_size)
        param.data.copy_(loaded_shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass:
        输入是完整的，输出是切分的。
        [B, S, H] -> [B, S, H/TP]
        """
        output_parallel = F.linear(x, self.weight, self.bias)
        return output_parallel


class RowParallelLinear(nn.Module):
    """
    行并行线性层 (Row Parallel Linear Layer).
    
    通常用于 Transformer MLP 的第二层 (Down Projection)。
    它接收被切分的输入 (Sharded Input)，并输出完整的张量 (Full Output)。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        process_group: dist.ProcessGroup = None,
    ):
        super().__init__()
        self.process_group = process_group
        self.tp_rank = dist.get_rank(group=process_group)
        self.tp_size = dist.get_world_size(group=process_group)

        # 1. 维度切分逻辑
        # Row Parallel 切分的是 输入维度 (input_size)
        assert input_size % self.tp_size == 0, f"Input size {input_size} must be divisible by TP size {self.tp_size}"
        
        self.input_size_per_partition = input_size // self.tp_size
        self.output_size = output_size

        # 2. 定义本地权重
        # PyTorch 的 nn.Linear 权重形状是 [out_features, in_features]
        # Row Parallel: [output_size, input_size // tp_size]
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        
        # 3. Bias 处理
        # Bias 不切分，每个 Rank 持有完整副本，但只在 Reduce 后加一次
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # Row Parallel 切分的是 in_features (dim=1)
        tp_dim = 1
        shard_size = self.input_size_per_partition
        start_idx = self.tp_rank * shard_size
        
        # 从完整权重中切出当前 Rank 负责的列
        loaded_shard = loaded_weight.narrow(tp_dim, start_idx, shard_size)
        param.data.copy_(loaded_shard)

    def bias_loader(self, param: nn.Parameter, loaded_bias: torch.Tensor):
        # Bias 不切分，直接拷贝
        param.data.copy_(loaded_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass:
        输入是切分的，输出是完整的。
        1. 本地计算: Y_partial = X_shard @ W_shard
        2. All-Reduce: Y = sum(Y_partial)
        3. Add Bias
        """
        output_parallel = F.linear(x, self.weight)

        if self.tp_size > 1:
            dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM, group=self.process_group)

        if self.bias is not None:
            output_parallel += self.bias

        return output_parallel