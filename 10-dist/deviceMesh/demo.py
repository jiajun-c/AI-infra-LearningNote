"""
DeviceMesh 2D 并行示例：DP + TP

启动方式（8 卡）：
    torchrun --nproc_per_node=8 demo.py

Mesh 结构：
    mesh_shape=(2, 4), dim_names=("dp", "tp")

    物理 GPU 映射为 2×4 网格：
         tp=0   tp=1   tp=2   tp=3
    dp=0  gpu0   gpu1   gpu2   gpu3
    dp=1  gpu4   gpu5   gpu6   gpu7

    mesh["dp"] → 每个 dp 组内的 4 个 tp rank（TP 通信组）
    mesh["tp"] → 每个 tp 组内的 2 个 dp rank（DP 通信组）
"""

import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def demo_mesh_structure(mesh):
    """展示 DeviceMesh 的结构和进程组关系"""
    rank = dist.get_rank()

    if rank == 0:
        print(f"\n=== Mesh 结构 ===")
        print(f"  mesh.shape       = {mesh.shape}")
        print(f"  mesh.mesh        = {mesh.mesh}")
        print(f"  mesh_dim_names   = {mesh.mesh_dim_names}")
        print(f"  mesh[\"dp\"] shape  = {mesh['dp'].shape}  (TP 通信组: {mesh['dp'].mesh})")
        print(f"  mesh[\"tp\"] shape  = {mesh['tp'].shape}  (DP 通信组: {mesh['tp'].mesh})")

    dist.barrier()

    # 通过 get_group 获取 NCCL 通信组
    tp_group = mesh.get_group("tp")
    dp_group = mesh.get_group("dp")

    # 验证 TP 组内的 rank 布局
    tp_submesh = mesh["dp"]  # 同一个 dp，所有 tp rank
    dp_submesh = mesh["tp"]  # 同一个 tp，所有 dp rank

    print(f"[rank {rank:1d}] "
          f"tp_local_rank = {tp_submesh.get_local_rank():1d}, "
          f"dp_local_rank = {dp_submesh.get_local_rank():1d}, "
          f"coordinate = {mesh.get_coordinate()}")

    return tp_group, dp_group, tp_submesh, dp_submesh


def demo_all_reduce(mesh, tp_group, dp_group):
    """演示在 mesh 的不同维度上做通信"""
    rank = dist.get_rank()

    # TP 组内 AllReduce（模拟列并行 linear 后的归约）
    tp_tensor = torch.ones(1).cuda() * rank
    dist.all_reduce(tp_tensor, group=tp_group)
    expected_tp = sum(r for r in range(rank, rank + 1) if r // 4 == rank // 4)  # 简单验证

    # DP 组内 AllReduce（梯度同步）
    dp_tensor = torch.ones(1).cuda() * rank
    dist.all_reduce(dp_tensor, group=dp_group)

    if rank == 0:
        print(f"\n=== 通信验证 ===")
        print(f"  TP all_reduce: 每个 rank 的 tensor={rank} → 组内求和 (同 dp，跨 tp)")
        print(f"  DP all_reduce: 每个 rank 的 tensor={rank} → 组内求和 (同 tp，跨 dp)")


def demo_column_parallel_linear(mesh, tp_group, tp_submesh):
    """
    列并行 Linear：按列切分 W，输出在 TP 组内做 AllReduce

    Y = X @ W
    W: (in_features, out_features) → 按 out_features 切分为 (in_features, out_features/4)
    每个 tp rank 持有 W 的 1/4 列
    """
    rank = dist.get_rank()
    tp_rank = tp_submesh.get_local_rank()
    tp_size = tp_submesh.size()

    batch, in_dim, out_dim = 4, 8, 16
    local_out_dim = out_dim // tp_size

    # 全量 W (只在 rank 0 打印)
    torch.manual_seed(42)
    W_full = torch.randn(in_dim, out_dim).cuda()

    # 每个 tp rank 持有 W 的 1/tp_size 列
    W_local = W_full[:, tp_rank * local_out_dim:(tp_rank + 1) * local_out_dim].clone()

    # 输入 X 在 dp 维度分割（每个 dp 组有不同的 batch 数据）
    X = torch.randn(batch, in_dim).cuda()

    # 本地计算：X @ W_local (batch, local_out_dim)
    Y_local = X @ W_local

    # TP 组内 AllReduce 得到完整输出
    Y = Y_local.clone()
    dist.all_reduce(Y, group=tp_group)

    # 验证：与全量 W 的计算结果对比
    Y_ref = X @ W_full
    assert torch.allclose(Y, Y_ref, atol=1e-5), f"rank {rank}: column parallel linear mismatch!"

    if rank == 0:
        print(f"\n=== 列并行 Linear 验证 ===")
        print(f"  W: {in_dim}×{out_dim} → 每 rank 持有 {in_dim}×{local_out_dim}")
        print(f"  X: {batch}×{in_dim} (dp 维度可能有不同 batch)")
        print(f"  Y_local + AllReduce → Y  {Y.shape}")
        print(f"  ✓ 与全量 W 计算结果一致")


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    # 2D mesh: 2 个 dp 组 × 4 个 tp 组 = 8 GPU
    mesh = init_device_mesh(
        "cuda",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp"),
    )

    if rank == 0:
        print("=" * 50)
        print(f"DeviceMesh DP×TP = 2×4 (共 {dist.get_world_size()} GPU)")
        print("=" * 50)

    # 1. 展示 mesh 结构
    tp_group, dp_group, tp_submesh, dp_submesh = demo_mesh_structure(mesh)

    # 2. 通信验证
    demo_all_reduce(mesh, tp_group, dp_group)

    # 3. 列并行 Linear
    demo_column_parallel_linear(mesh, tp_group, tp_submesh)

    dist.barrier()
    if rank == 0:
        print(f"\n=== 全部通过 ===")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
