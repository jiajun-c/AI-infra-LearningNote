import argparse
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

def all_gather(shard, world):
    parts = [torch.empty_like(shard) for _ in range(world)]
    dist.all_gather(parts, shard.contiguous())
    return torch.cat(parts)

def reduce_scatter(grad, world):
    pieces = list(grad.contiguous().chunk(world))
    result = torch.empty_like(pieces[0])
    dist.reduce_scatter(result, pieces, op=dist.ReduceOp.SUM)
    return result / world

class ShardedAdam:
    def __init__(self, parameters, lr=1e-3):
        self.parameters = parameters.detach().clone()
        self.m = torch.zeros_like(self.parameters)
        self.v = torch.zeros_like(self.parameters)
        self.lr = lr
        self.t = 0

    @torch.no_grad
    def step(self, grad):
        self.t += 1
        self.m.mul_(0.9).add_(grad, alpha=0.1)
        self.v.mul_(0.999).addcmul_(grad, grad, value=0.001)
        m_hat = self.m / (1 - 0.9 ** self.t)
        v_hat = self.v / (1 - 0.999 ** self.t)
        self.parameters.addcdiv_(m_hat, v_hat.sqrt().add(1e-8), value=-self.lr)


class ZeRO1:
    def __init__(self, weights, rank, world, lr=1e-3):
        self.rank, self.world = rank, world
        self.shapes = [w.shape for w in weights]
        self.sizes = [w.numel() for w in weights]
        self.flat = torch.cat([w.detach().flatten() for w in weights])
        self.optimizer = ShardedAdam(self.flat.chunk(world)[rank].contiguous(), lr)
        self.parameters = None
        self.loss = None
        self.gradient = None

    def forward(self, x, target):
        self.parameters = []
        offset = 0
        for shape, size in zip(self.shapes, self.sizes):
            p = self.flat[offset: offset+size].view(shape).detach().requires_grad_()
            self.parameters.append(p)
            offset += size
        y = (x @ self.parameters[0]) @ self.parameters[1]
        self.loss = F.mse_loss(y, target)
        return self.loss

    def backward(self):
        self.loss.backward()
        gradient = torch.cat([p.grad.flatten() for p in self.parameters])
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        gradient /= self.world
        self.gradient = gradient.chunk(self.world)[self.rank].contiguous()
        self.parameters = None

    def step(self):
        self.optimizer.step(self.gradient)
        self.flat = all_gather(self.optimizer.parameters, self.world)
        self.gradient = None

class ZeRO2(ZeRO1):
    def backward(self):
        self.loss.backward()
        gradient = torch.cat([p.grad.flatten() for p in self.parameters])
        self.gradient = reduce_scatter(gradient, self.world)
        self.parameters = None


class _ShardedMatmul(torch.autograd.Function):
    """Gather one weight for compute, then release it until backward."""

    @staticmethod
    def forward(ctx, x, shard, shape, numel, world):
        weight = all_gather(shard, world)[:numel].view(shape)
        ctx.save_for_backward(x, shard)
        ctx.shape = shape
        ctx.numel = numel
        ctx.world = world
        return x @ weight

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, shard = ctx.saved_tensors
        # The full weight was not saved in forward, so gather it again to
        # propagate gradients to the preceding layer.
        weight = all_gather(shard, ctx.world)[:ctx.numel].view(ctx.shape)
        grad_x = grad_output @ weight.T
        grad_weight = x.T @ grad_output
        padded_grad = F.pad(
            grad_weight.flatten(),
            (0, shard.numel() * ctx.world - ctx.numel),
        )
        grad_shard = reduce_scatter(padded_grad, ctx.world)
        return grad_x, grad_shard, None, None, None


class ZeRO3:
    """Two-matrix ZeRO-3 example with sharded model and FP32 master weights.

    Every rank must call forward, backward, and step in the same order.
    Full input weights are read during construction; afterward this object
    keeps only this rank's weight, gradient, and optimizer-state shards.
    Callers must release their original full weights to realize that saving.
    """

    def __init__(self, weights, rank, world, lr=1e-3, model_dtype=None):
        weights = list(weights)
        if len(weights) != 2 or any(w.ndim != 2 for w in weights):
            raise ValueError("ZeRO3 expects two 2D weight matrices")
        if not dist.is_initialized():
            raise RuntimeError("initialize the distributed process group first")
        if rank != dist.get_rank() or world != dist.get_world_size():
            raise ValueError("rank and world must match the process group")

        self.rank, self.world = rank, world
        self.shapes = [w.shape for w in weights]
        self.sizes = [w.numel() for w in weights]
        self.model_dtype = model_dtype or weights[0].dtype
        if self.model_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("model_dtype must be float16, bfloat16, or float32")

        self.model_shards = []
        self.optimizers = []
        for weight, size in zip(weights, self.sizes):
            shard_size = (size + world - 1) // world
            flat_fp32 = weight.detach().flatten().float()
            padded = F.pad(flat_fp32, (0, shard_size * world - size))
            master_shard = padded[rank * shard_size:(rank + 1) * shard_size].clone()
            optimizer = ShardedAdam(master_shard, lr)
            model_shard = optimizer.parameters.to(self.model_dtype).clone().detach()
            model_shard.requires_grad_()
            self.optimizers.append(optimizer)
            self.model_shards.append(model_shard)

        self.loss = None
        self.gradients = None

    def forward(self, x, target):
        if self.loss is not None or self.gradients is not None:
            raise RuntimeError("finish backward and step before the next forward")
        x = x.to(self.model_dtype)
        hidden = _ShardedMatmul.apply(
            x, self.model_shards[0], self.shapes[0], self.sizes[0], self.world
        )
        output = _ShardedMatmul.apply(
            hidden, self.model_shards[1], self.shapes[1], self.sizes[1], self.world
        )
        # Compute MSE in FP32 even when the matrix multiplications use FP16.
        self.loss = F.mse_loss(output.float(), target.float())
        return self.loss

    def backward(self):
        if self.loss is None:
            raise RuntimeError("call forward before backward")
        self.loss.backward()
        self.gradients = []
        for shard in self.model_shards:
            if shard.grad is None:
                raise RuntimeError("a model shard did not receive a gradient")
            self.gradients.append(shard.grad)
            shard.grad = None
        self.loss = None

    @torch.no_grad()
    def step(self):
        if self.gradients is None:
            raise RuntimeError("call backward before step")
        for shard, optimizer, grad in zip(
            self.model_shards, self.optimizers, self.gradients
        ):
            optimizer.step(grad.float())
            shard.copy_(optimizer.parameters.to(shard.dtype))
        self.gradients = None

