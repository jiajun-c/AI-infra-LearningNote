import torch
import torch.distributed as dist


def distMyTrans(x):
    world_size = dist.get_world_size()

    x_t = x.t().contiguous()            # [N, local_rows]
    x_new = torch.zeros_like(x_t)
    dist.all_to_all_single(x_new, x_t)  # 按 dim=0 均分交换，结果仍是 [N, local_rows]

    # x_new 中 W 块 [N/W, local_rows] 沿 dim=0 拼接，需转为沿 dim=1 拼接得 [N/W, M]
    N = x_new.shape[0]
    chunk_size = N // world_size
    return torch.cat(x_new.split(chunk_size, dim=0), dim=1)  # [N/W, M]
    

def distTrans(x):
    """
    分布式转置：每个rank持有 [M/W, N] 的行分片，转置后持有 [N/W, M] 的列分片。

    输入: x shape = [local_rows, N]，其中 local_rows = M / world_size
    输出: shape = [N / world_size, M]

    实现步骤:
    1. 本地转置: [local_rows, N] -> [N, local_rows]
    2. 按 world_size 分块: N 维切成 W 块，每块 [N/W, local_rows]
    3. all_to_all: rank_i 收到所有rank贡献的 [N/W, local_rows_i] 块
    4. 拼接列维: W 块 [N/W, local_rows] -> [N/W, M]
    """
    world_size = dist.get_world_size()

    # step1: 本地转置
    x_t = x.t().contiguous()  # [N, local_rows]

    N = x_t.shape[0]
    assert N % world_size == 0, "N must be divisible by world_size"

    # step2: 切块，准备 all_to_all 的发送列表
    chunk_size = N // world_size
    send_list = list(x_t.split(chunk_size, dim=0))  # W x [N/W, local_rows]

    # step3: all_to_all 交换
    recv_list = [torch.empty_like(send_list[0]) for _ in range(world_size)]
    dist.all_to_all(recv_list, send_list)

    # step4: 沿列维拼接
    return torch.cat(recv_list, dim=1)  # [N/W, M]


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    M, N = 128, 1024  # 全局矩阵形状
    local_rows = M // world_size

    # 每个 rank 持有全局矩阵的一段行，值填入 rank 编号便于验证
    x = torch.full((local_rows, N), float(rank), device=device)

    y = distTrans(x)    # [N/W, M]
    y2 = distMyTrans(x) # [N/W, M]

    # 验证：每个rank独立验证自己持有的列分片
    # rank_r 应持有全局转置矩阵第 [r*N//W : (r+1)*N//W] 行
    # 全局矩阵 full_x[i, :] = float(i // local_rows)，其转置 full_xt[:, i] = float(i // local_rows)
    # 因此 expected[j, i] = float(i // local_rows)，i in [0, M), j in [rank*chunk : (rank+1)*chunk)
    chunk = N // world_size
    col_idx = torch.arange(M, device=device)                          # [M]
    expected_row = (col_idx // local_rows).float()                    # [M]
    expected = expected_row.unsqueeze(0).expand(chunk, -1)            # [N/W, M]

    assert y.shape == expected.shape, f"shape不匹配: {y.shape} vs {expected.shape}"
    assert torch.allclose(y, expected), f"rank {rank} distTrans 数值不匹配!\n得到:\n{y}\n期望:\n{expected}"
    print(f"rank {rank} ✓ distTrans 验证通过，本地输出形状: {y.shape}")

    assert y2.shape == expected.shape, f"shape不匹配: {y2.shape} vs {expected.shape}"
    assert torch.allclose(y2, expected), f"rank {rank} distMyTrans 数值不匹配!\n得到:\n{y2}\n期望:\n{expected}"
    print(f"rank {rank} ✓ distMyTrans 验证通过，本地输出形状: {y2.shape}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
