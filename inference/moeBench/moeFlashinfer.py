import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入 FlashInfer
from flashinfer.fused_moe import cutlass_fused_moe

# ==========================================
# 路由门控模块 (保持不变，但强调 Float32 输出)
# ==========================================
class TopKRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.classifier = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x):
        logits = self.classifier(x)
        routing_probs = F.softmax(logits, dim=-1)
        routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # 🚨 极其关键：FlashInfer 强制要求 routing_weights 是 Float32 格式，不可用半精度！
        routing_weights = routing_weights.to(torch.float32)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, selected_experts


# ==========================================
# 🌟 FlashInfer MoE 核心层 (重构版)
# ==========================================
class FlashInferMoELayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = TopKRouter(hidden_size, num_experts, top_k)
        
        # 🚨 核心改造：抛弃 nn.ModuleList！
        # 将所有专家的权重合并为一整块连续的 Parameter 张量。
        # CUTLASS 严格要求的形状是 [num_experts, out_features, in_features]
        
        # w1 包含了 Gate 和 Up 两个矩阵，输出维度是 intermediate_size * 2
        self.w1 = nn.Parameter(
            torch.empty(num_experts, intermediate_size * 2, hidden_size)
        )
        
        # w2 是 Down 矩阵，输出维度是 hidden_size，输入是 intermediate_size
        self.w2 = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        
        self._init_weights()

    def _init_weights(self):
        # 采用我们在 Benchmark 中验证过的安全缩放，防止 NaN 溢出
        nn.init.normal_(self.w1, mean=0.0, std=0.01)
        nn.init.normal_(self.w2, mean=0.0, std=0.01)

    def forward(self, x):
        """
        x: [batch_size, seq_len, hidden_size] (必须是 Float16 或 BFloat16)
        """
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        # 1. 获取路由权重和索引
        # routing_weights: [num_tokens, top_k] (Float32)
        # selected_experts: [num_tokens, top_k] (Int64)
        routing_weights, selected_experts = self.router(x_flat)
        
        # 2. 调用 FlashInfer 高度融合的 CUTLASS 算子
        result = cutlass_fused_moe(
            input=x_flat,
            # 强制转换为 int32 并保证物理显存连续
            token_selected_experts=selected_experts.int().contiguous(),
            # 保证物理显存连续
            token_final_scales=routing_weights.contiguous(),
            fc1_expert_weights=self.w1,
            fc2_expert_weights=self.w2,
            output_dtype=x_flat.dtype,
            quant_scales=None  # 绕过最新版 API 的量化参数检查
        )
        
        # 3. 提取结果 (处理 FlashInfer 新版本返回 List 的坑)
        out_flat = result[0] if isinstance(result, (list, tuple)) else result
        
        # 4. 恢复原始形状
        return out_flat.view(batch_size, seq_len, hidden_size)


# ==========================================
# 快速验证脚本
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    batch_size = 2
    seq_len = 8
    hidden_size = 64
    intermediate_size = 128
    num_experts = 4
    top_k = 2

    # 创建 FP16 的输入
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    
    # 实例化搭载了 FlashInfer 引擎的 MoE 层
    moe_layer = FlashInferMoELayer(
        hidden_size, intermediate_size, num_experts, top_k
    ).to(device).to(dtype)
    
    # 执行前向传播
    output = moe_layer(x)
    
    print(f"输入形状: {x.shape} | 数据类型: {x.dtype}")
    print(f"输出形状: {output.shape} | 数据类型: {output.dtype}")
    print("🚀 FlashInfer MoE 层前向传播成功！")