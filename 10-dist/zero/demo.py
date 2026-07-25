"""
ZeRO-2 风格梯度分片 demo

目标：
1. 模拟 4 个 data parallel rank 各自算出本地参数梯度
2. 对比 DDP 的 all-reduce 和 ZeRO-2 的 reduce-scatter
3. 展示“每个 rank 只保留自己的梯度分片，也能完成参数更新”

这个 demo 不依赖真实多进程，而是用普通 Python / PyTorch 张量
在一个进程里把 4 个 rank 的行为模拟出来，方便理解数据流。
"""

import torch


def chunk_tensor(x: torch.Tensor, world_size: int) -> list[torch.Tensor]:
    """把一维张量均匀切成 world_size 片。"""
    assert x.dim() == 1
    assert x.numel() % world_size == 0
    return list(torch.chunk(x, world_size))


def simulate_allreduce_sum(local_grads: list[torch.Tensor]) -> list[torch.Tensor]:
    """DDP 风格：sum 后每个 rank 都拿到完整梯度。"""
    full_grad = torch.stack(local_grads, dim=0).sum(dim=0)
    return [full_grad.clone() for _ in local_grads]


def simulate_reduce_scatter_sum(local_grads: list[torch.Tensor]) -> list[torch.Tensor]:
    """ZeRO-2 风格：先求和，再把完整梯度切片分发给各个 rank。"""
    world_size = len(local_grads)
    full_grad = torch.stack(local_grads, dim=0).sum(dim=0)
    shards = chunk_tensor(full_grad, world_size)
    return [shard.clone() for shard in shards]


def main() -> None:
    torch.manual_seed(0)

    world_size = 4
    lr = 0.1

    # 假设一个参数向量长度为 8，方便 4 卡均匀切成 4 片，每片长度 2
    full_param = torch.arange(1.0, 9.0)

    # 每个 rank 本地 batch 反向传播得到的“完整参数梯度”
    # 在真实 DDP / ZeRO 里，每个 rank 都会先算出对完整参数的本地梯度贡献
    local_grads = [
        torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 2.0, 2.0]),
        torch.tensor([0.5, 1.0, 1.5, 2.0, 0.5, 0.5, 1.0, 1.0]),
        torch.tensor([2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0]),
        torch.tensor([1.5, 1.0, 0.5, 1.0, 1.5, 1.0, 0.5, 1.0]),
    ]

    print("=== Initial Parameter ===")
    print(f"full_param = {full_param.tolist()}")
    print()

    print("=== Local Gradients From Backward ===")
    for rank, grad in enumerate(local_grads):
        print(f"rank{rank}: {grad.tolist()}")
    print()

    # 1) DDP: all-reduce 后每个 rank 都拿到完整梯度
    ddp_full_grads = simulate_allreduce_sum(local_grads)
    ddp_global_grad = ddp_full_grads[0]

    print("=== DDP: AllReduce Result ===")
    print(f"global_grad = {ddp_global_grad.tolist()}")
    print("each rank keeps the full gradient")
    for rank, grad in enumerate(ddp_full_grads):
        print(f"rank{rank}: grad_numel = {grad.numel()}, grad = {grad.tolist()}")
    print()

    # 2) ZeRO-2: reduce-scatter 后每个 rank 只拿到自己那一片梯度
    zero2_grad_shards = simulate_reduce_scatter_sum(local_grads)
    param_shards = chunk_tensor(full_param, world_size)

    print("=== ZeRO-2: ReduceScatter Result ===")
    for rank, shard in enumerate(zero2_grad_shards):
        print(f"rank{rank}: grad_shard_numel = {shard.numel()}, grad_shard = {shard.tolist()}")
    print()

    # 3) 每个 rank 只更新自己的参数分片
    updated_param_shards = []
    print("=== Per-Rank Sharded Optimizer Step ===")
    for rank, (param_shard, grad_shard) in enumerate(zip(param_shards, zero2_grad_shards)):
        updated = param_shard - lr * grad_shard
        updated_param_shards.append(updated)
        print(
            f"rank{rank}: param_shard = {param_shard.tolist()}, "
            f"updated_shard = {updated.tolist()}"
        )
    print()

    zero2_updated_full_param = torch.cat(updated_param_shards, dim=0)
    ddp_updated_full_param = full_param - lr * ddp_global_grad

    print("=== Compare Final Parameters ===")
    print(f"DDP full update     = {ddp_updated_full_param.tolist()}")
    print(f"ZeRO-2 shard update = {zero2_updated_full_param.tolist()}")
    print(f"match = {torch.allclose(ddp_updated_full_param, zero2_updated_full_param)}")
    print()

    print("=== Takeaway ===")
    print("1. backward 之后，每个 rank 先拿到的是本地 batch 对完整参数的梯度贡献")
    print("2. DDP 用 all-reduce，让每个 rank 都保留完整梯度")
    print("3. ZeRO-2 用 reduce-scatter，让每个 rank 只保留自己的梯度分片")
    print("4. 只要参数分片、梯度分片、优化器状态分片对齐，就能正确更新")


if __name__ == "__main__":
    main()
