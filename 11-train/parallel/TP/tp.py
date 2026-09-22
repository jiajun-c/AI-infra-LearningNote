"""Megatron-style 1D tensor parallelism implemented with torch.distributed.

Run the correctness example with:

    torchrun --standalone --nproc-per-node=2 tp.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _tp_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _tp_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _ensure_divisible(value: int, divisor: int, name: str) -> None:
    if value % divisor:
        raise ValueError(f"{name}={value} must be divisible by TP size={divisor}")


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """Forward: identity; backward: all-reduce dX."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        if _tp_world_size() > 1:
            dist.all_reduce(grad_output)
        return grad_output


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """Forward: all-reduce partial outputs; backward: identity."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        if _tp_world_size() > 1:
            x = x.clone()  # Do not overwrite an activation used elsewhere.
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class _ScatterToTensorParallelRegion(torch.autograd.Function):
    """Forward: split the last dimension; backward: concatenate gradients."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        world_size = _tp_world_size()
        _ensure_divisible(x.size(-1), world_size, "input hidden size")
        return x.chunk(world_size, dim=-1)[_tp_rank()].contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        if _tp_world_size() == 1:
            return grad_output
        parts = [torch.empty_like(grad_output) for _ in range(_tp_world_size())]
        dist.all_gather(parts, grad_output.contiguous())
        return torch.cat(parts, dim=-1)


class _GatherFromTensorParallelRegion(torch.autograd.Function):
    """Forward: concatenate shards; backward: select this rank's shard."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        if _tp_world_size() == 1:
            return x
        parts = [torch.empty_like(x) for _ in range(_tp_world_size())]
        dist.all_gather(parts, x.contiguous())
        return torch.cat(parts, dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.chunk(_tp_world_size(), dim=-1)[_tp_rank()].contiguous()


class ColumnParallelLinear(nn.Module):
    """Shard ``weight[out_features, in_features]`` over output features.

    All ranks receive the same input. The normal forward path has no
    communication; the autograd copy all-reduces dX during backward.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        world_size = _tp_world_size()
        _ensure_divisible(out_features, world_size, "out_features")
        self.in_features = in_features
        self.out_features = out_features
        self.output_size_per_partition = out_features // world_size
        self.gather_output = gather_output
        kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, in_features, **kwargs)
        )
        self.bias = (
            nn.Parameter(torch.empty(self.output_size_per_partition, **kwargs))
            if bias
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -self.in_features**-0.5, self.in_features**-0.5)

    @torch.no_grad()
    def load_from_linear(self, linear: nn.Linear) -> None:
        self.weight.copy_(linear.weight.chunk(_tp_world_size(), dim=0)[_tp_rank()])
        if self.bias is not None and linear.bias is not None:
            self.bias.copy_(linear.bias.chunk(_tp_world_size(), dim=0)[_tp_rank()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _CopyToTensorParallelRegion.apply(x)
        output = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            output = _GatherFromTensorParallelRegion.apply(output)
        return output


class RowParallelLinear(nn.Module):
    """Shard ``weight[out_features, in_features]`` over input features.

    Each rank computes a partial result and forward all-reduces it. Bias is
    replicated, so it is added exactly once, after the all-reduce.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        world_size = _tp_world_size()
        _ensure_divisible(in_features, world_size, "in_features")
        self.in_features = in_features
        self.out_features = out_features
        self.input_size_per_partition = in_features // world_size
        self.input_is_parallel = input_is_parallel
        kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(out_features, self.input_size_per_partition, **kwargs)
        )
        self.bias = nn.Parameter(torch.empty(out_features, **kwargs)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -self.in_features**-0.5, self.in_features**-0.5)

    @torch.no_grad()
    def load_from_linear(self, linear: nn.Linear) -> None:
        self.weight.copy_(linear.weight.chunk(_tp_world_size(), dim=1)[_tp_rank()])
        if self.bias is not None and linear.bias is not None:
            self.bias.copy_(linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            x = _ScatterToTensorParallelRegion.apply(x)
        output_parallel = F.linear(x, self.weight, bias=None)
        output = _ReduceFromTensorParallelRegion.apply(output_parallel)
        return output if self.bias is None else output + self.bias


class TensorParallelSwiGLU(nn.Module):
    """Fused gate/up column parallel + down row parallel LLaMA MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size, 2 * intermediate_size, bias, False, device, dtype
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias, True, device, dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def _correctness_example(device: torch.device) -> None:
    """Compare the TP MLP and an equivalent dense MLP, including dX."""
    torch.manual_seed(1234)
    hidden_size, intermediate_size = 16, 32
    gate = nn.Linear(hidden_size, intermediate_size, bias=False, device=device)
    up = nn.Linear(hidden_size, intermediate_size, bias=False, device=device)
    down = nn.Linear(intermediate_size, hidden_size, bias=False, device=device)

    # Layout the fused weight as [rank0 gate, rank0 up, rank1 gate, rank1 up].
    fused = nn.Linear(hidden_size, 2 * intermediate_size, bias=False, device=device)
    with torch.no_grad():
        shards = []
        for gate_part, up_part in zip(
            gate.weight.chunk(_tp_world_size(), dim=0),
            up.weight.chunk(_tp_world_size(), dim=0),
        ):
            shards.extend((gate_part, up_part))
        fused.weight.copy_(torch.cat(shards, dim=0))

    tp_mlp = TensorParallelSwiGLU(hidden_size, intermediate_size, device=device)
    tp_mlp.gate_up_proj.load_from_linear(fused)
    tp_mlp.down_proj.load_from_linear(down)

    x = torch.randn(2, 4, hidden_size, device=device, requires_grad=True)
    dense_x = x.detach().clone().requires_grad_(True)
    actual = tp_mlp(x)
    expected = down(F.silu(gate(dense_x)) * up(dense_x))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(x.grad, dense_x.grad, rtol=1e-5, atol=1e-6)
    if _tp_rank() == 0:
        print(f"TP={_tp_world_size()}: forward and backward match dense MLP")


def main() -> None:
    if "RANK" in os.environ:
        use_cuda = torch.cuda.is_available()
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
        if use_cuda:
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _correctness_example(device)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
