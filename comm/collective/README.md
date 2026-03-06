# 集合通信原语

## 1. All-Gather

All-gather将多个节点上的部分数据聚合为一个节点上的完整数据，最终所有节点都有一份完整数据的副本。

操作逻辑：节点按照逻辑换换顺序将数据发送给下一个节点，下一个节点将数据接收，并保存在接收节点的缓存中，接收节点将数据发送给下一个节点，直到所有节点都完成数据发送。

## 2. Reduce-Scatter

Reduce-Scatter 先对多点进行规约的操作，然后将结果分散到各节点上，最终每个节点仅保留部分规约结果。

## 3. All-Reduce

All-Reduce 将所有节点的数据进行求和的操作，然后将数据发送到所有的节点上。

如下所示，使用三个节点的
```python3
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")

if __name__ == "__main__":
    world_size = 4
    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)
```