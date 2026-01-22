import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import os
# 引入 Profiler 相关模块
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert output_size % self.tp_size == 0
        
        self.tp_out_size = output_size // self.tp_size
        self.weight = nn.Parameter(torch.empty(self.tp_out_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.tp_out_size))
        else:
            self.register_parameter("bias", None)
    
    def weight_loader(self, weight: torch.Tensor):
        tp_dim = 0
        start_idx = self.tp_out_size * self.tp_rank
        # weight 在 CPU，narrow 也在 CPU，copy_ 会处理 host->device
        load_weight = weight.narrow(tp_dim, start_idx, self.tp_out_size)
        self.weight.data.copy_(load_weight)
        
    def forward(self, x: torch.Tensor):
        output_parallel = F.linear(x, self.weight, self.bias)
        # print(output_parallel) # 移除 print 以免干扰 profile
        return output_parallel

def demo(rank, world_size):
    torch.manual_seed(123)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'
    
    # 1. 初始化
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    print(f"[Rank {rank}] Initialized on {device}")

    # 2. 准备模型和数据
    BATCH_SIZE = 2
    INPUT_SIZE = 4
    OUTPUT_SIZE = 8

    # 模型移动到 GPU
    layer = ColumnParallelLinear(INPUT_SIZE, OUTPUT_SIZE, bias=False).to(device)

    # 模拟加载权重 (full_weight 在 CPU)
    full_weight = torch.ones(OUTPUT_SIZE, INPUT_SIZE)
    layer.weight_loader(full_weight) 

    # 输入数据 (必须在 GPU)
    input_data = torch.ones(BATCH_SIZE, INPUT_SIZE).to(device)
    
    # 计算 Golden 结果 (不带 bias)
    golden_out = F.linear(input_data, full_weight.to(device))

    # 3. 前向传播
    output = layer(input_data) 
    
    # 4. 准备 Gather (Zero-Copy 方案)
    fullout = None
    gather_list = None

    if rank == 0:
        # 1. 预分配 Buffer: 形状 [TP_SIZE, BATCH, LOCAL_HIDDEN]
        local_hidden = OUTPUT_SIZE // world_size
        gather_buffer = torch.empty(world_size, BATCH_SIZE, local_hidden, device=device, dtype=output.dtype)
        
        # 2. 创建 gather_list (指向 buffer 的连续分片)
        gather_list = [gather_buffer[i] for i in range(world_size)]
    else:
        gather_list = None
    
    # --- Profiling 开始 ---
    # 预热一次 (Warmup)
    dist.gather(output, gather_list=gather_list, dst=0)
    
    # 创建日志目录
    log_dir = "./log"
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)

    # 开启 Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=None, 
        on_trace_ready=tensorboard_trace_handler(f"{log_dir}/rank_{rank}"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        # 5. 执行 Gather
        with record_function("dist_gather"):
            dist.gather(output, gather_list=gather_list, dst=0)

        # 6. 验证与后处理
        if rank == 0:
            with record_function("concat_step"):
                fullout = torch.cat(gather_list, dim=1)

    # --- Profiling 结束 ---

    if rank == 0:
        print(f"\n[Rank 0] Profiling 完成。日志已保存至 {log_dir}/rank_{rank}")
        # 打印主要的耗时操作
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        print(f"\nfullout shape: {fullout.shape}")
        
        # 验证误差
        diff = (fullout - golden_out).abs().max()
        print(f"最大误差 (Max Diff): {diff.item():.6f}")
        
        if diff < 1e-5:
            print(">>> 验证成功！结果一致。")
        else:
            print(">>> 验证失败！误差过大。")
            print("Gathered:", fullout.tolist())
            print("Golden:  ", golden_out.tolist())

    dist.destroy_process_group()

if __name__ == "__main__":
    WORLD_SIZE = 2
    mp.spawn(demo, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)