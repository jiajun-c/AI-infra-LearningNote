# torch 使用共享内存接口

torch中提供了对称内存的编程接口，其大致流程如下

申请一块显存区域用于共享内存通信

```python
t = symm_mem.empty((1024,), dtype=torch.float32, device=device)
```

然后建立虚拟地址的映射

```python
hdl = symm_mem.rendezvous(t, dist.group.WORLD)
```

获取这个远程的指针

```python
neighbor_t = hdl.get_buffer(neighbor_rank, t.shape, t.dtype)
```

对远程的数据进行Load和Store

```python
    # 直接读取 (Load 动作)
    val = neighbor_t[0].item()
    print(f"[Rank {rank}] 直接从邻居卡 {neighbor_rank} 读取到了值: {val}\n")

    # 7. 远程写入 (Store 动作)
    with torch.no_grad():
        # 修改邻居卡上的第 rank 个位置
        neighbor_t[rank] = rank * 100.0
    dist.barrier()
```

完整的样例代码

```python
import os
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

def main():
    # 1. 自动从 torchrun 获取环境变量
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # 2. 初始化
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 3. 申请对称内存 (1024个FP32元素)
    # 每个 rank 申请的空间大小必须一致
    t = symm_mem.empty((1024,), dtype=torch.float32, device=device)
    t.fill_(float(rank)) # 用 rank 填充，方便验证

    # 4. 建立虚拟地址映射 (建立跨卡“传送门”)
    # 注意：在单机多卡下使用 dist.group.WORLD
    hdl = symm_mem.rendezvous(t, dist.group.WORLD)

    # 5. 等待所有卡准备就绪
    dist.barrier()

    # 6. 读取邻居数据 (验证逻辑)
    neighbor_rank = (rank + 1) % world_size
    # 获取指向邻居显存的“远程指针” (Tensor 视图)
    neighbor_t = hdl.get_buffer(neighbor_rank, t.shape, t.dtype)

    # 直接读取 (Load 动作)
    val = neighbor_t[0].item()
    print(f"[Rank {rank}] 直接从邻居卡 {neighbor_rank} 读取到了值: {val}\n")

    # 7. 远程写入 (Store 动作)
    with torch.no_grad():
        # 修改邻居卡上的第 rank 个位置
        neighbor_t[rank] = rank * 100.0

    dist.barrier()

    # 8. 检查自己的内存是否被邻居改写了
    my_val = t[(rank - 1) % world_size].item()
    print(f"[Rank {rank}] 检查本地内存，发现被邻居改写为了: {my_val}\n")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```