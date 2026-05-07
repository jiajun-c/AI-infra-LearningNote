import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# 1. 显存监控探针 (Context Manager)
class MemoryTracker:
    def __init__(self, phase_name):
        self.phase_name = phase_name

    def __enter__(self):
        # 每次测量前清空缓存，重置峰值记录
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.start_mem = torch.cuda.memory_allocated() / (1024 ** 2) # 转换为 MB
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        self.peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        print(f"=== {self.phase_name} ===")
        print(f"当前常驻显存: {self.end_mem:.2f} MB")
        print(f"峰值占用显存: {self.peak_mem:.2f} MB")
        print(f"计算过程额外消耗峰值: {self.peak_mem - self.start_mem:.2f} MB\n")


# 2. 定义模型层（和之前一样）
class DummyBlock(nn.Module):
    def __init__(self, hidden_size=4096):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))

class LLMModel(nn.Module):
    def __init__(self, hidden_size=1024, num_layers=16, use_checkpointing=False):
        super().__init__()
        self.layers = nn.ModuleList([DummyBlock(hidden_size) for _ in range(num_layers)])
        self.use_checkpointing = use_checkpointing

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpointing:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x

# ================= 运行对比 =================
if __name__ == "__main__":
    # 为了防止显存爆炸，我们适当调小参数，但足以看出差距
    batch_size = 4
    seq_len = 1024
    hidden_dim = 1024
    
    print("初始化输入张量...")
    dummy_input = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True).cuda()
    dummy_target = torch.randn(batch_size, seq_len, hidden_dim).cuda() # 用于算 Loss
    
    # ---------------- 场景 A：标准训练 ----------------
    model_standard = LLMModel(hidden_size=hidden_dim, use_checkpointing=False).cuda()
    
    with MemoryTracker("标准模式 - 前向与反向传播"):
        out_standard = model_standard(dummy_input)
        loss_standard = ((out_standard - dummy_target)**2).mean()
        loss_standard.backward()
    
    # 清理掉标准模式占用的显存，为测试腾出空间
    del model_standard, out_standard, loss_standard
    torch.cuda.empty_cache()
    
    # ---------------- 场景 B：重计算训练 ----------------
    model_ckpt = LLMModel(hidden_size=hidden_dim, use_checkpointing=True).cuda()
    
    # 注意：重计算时输入张量的 requires_grad 必须为 True
    with MemoryTracker("重计算模式 - 前向与反向传播"):
        out_ckpt = model_ckpt(dummy_input)
        loss_ckpt = ((out_ckpt - dummy_target)**2).mean()
        loss_ckpt.backward()