import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    BATCH = 128
    SEQ_LEN = 256
    HIDDEN_DIM = 1024
    HIDDEN_OUT = 1024

    torch.manual_seed(42)
    # DP: 每卡处理 BATCH//world_size 条样本，权重各卡完整副本
    x_full = torch.ones([BATCH, SEQ_LEN, HIDDEN_DIM], dtype=torch.float32)
    w = torch.full([HIDDEN_OUT, HIDDEN_DIM], fill_value=0.01, dtype=torch.float32)

    # 按 batch 维切分给当前 rank
    local_batch = BATCH // world_size
    x_local = x_full[rank * local_batch: (rank + 1) * local_batch]  # [local_batch, S, H]

    # 本地 Linear 前向
    y_local = x_local @ w.T  # [local_batch, S, H_out]

    # --- 正确性验证 ---
    # 1. 单卡参考结果
    y_ref = x_full @ w.T  # [BATCH, S, H_out]

    # 2. 收集所有卡的输出，拼接后与参考结果对比
    gathered = [torch.zeros_like(y_local) for _ in range(world_size)]
    dist.all_gather(gathered, y_local)
    y_gathered = torch.cat(gathered, dim=0)  # [BATCH, S, H_out]

    match = torch.allclose(y_gathered, y_ref, atol=1e-5)
    if rank == 0:
        print(f"shape check : y_local={list(y_local.shape)}, y_gathered={list(y_gathered.shape)}")
        print(f"correctness : {'PASS' if match else 'FAIL'}")
        print(f"max_diff    : {(y_gathered - y_ref).abs().max().item():.2e}")

    dist.destroy_process_group()
    return y_local


if __name__ == "__main__":
    main()
