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
    t.fill_(float(rank+1)) # 用 rank 填充，方便验证

    # 4. 建立虚拟地址映射 (建立跨卡“传送门”)
    # 注意：在单机多卡下使用 dist.group.WORLD
    hdl = symm_mem.rendezvous(t, dist.group.WORLD)

    dist.barrier()
    
    # 5. 等待所有卡准备就绪
    # torch.ops.symm_mem.one_shot_all_reduce(t, "sum", dist.group.WORLD)
    group_name = dist._get_process_group_name(dist.group.WORLD)
    res = torch.ops.symm_mem.one_shot_all_reduce(t, "sum", group_name)
    
    dist.barrier()
    print(f"rank t[0] {res[0]}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()