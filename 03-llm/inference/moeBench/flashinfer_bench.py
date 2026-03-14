import torch
import torch.nn as nn
import torch.nn.functional as F
from flashinfer.fused_moe import cutlass_fused_moe

# ==========================================
# 共享组件：路由门控模块
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
        
        # FlashInfer 强制要求 routing_weights 是 Float32
        routing_weights = routing_weights.to(torch.float32)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        return routing_weights, selected_experts


# ==========================================
# Baseline: 纯 PyTorch MoE 层
# ==========================================
class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        h = self.w1(x)
        gate, up = h.chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)

class PyTorchMoELayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList([
            SwiGLUExpert(hidden_size, intermediate_size) for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        routing_weights, selected_experts = self.router(x_flat)
        out_flat = torch.zeros_like(x_flat)
        print(f"batch {batch_size}")
        print(f"selected_experts {selected_experts.shape}")
        for expert_idx in range(self.num_experts):
            token_mask = selected_experts == expert_idx
            token_idx, k_idx = token_mask.nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            
            x_expert = x_flat[token_idx]
            expert_out = self.experts[expert_idx](x_expert)
            g_i = routing_weights[token_idx, k_idx].unsqueeze(-1)
            expert_out = expert_out * g_i
            
            out_flat.index_add_(0, token_idx, expert_out.to(out_flat.dtype))
            
        return out_flat.view(batch_size, seq_len, hidden_size)


# ==========================================
# Optimized: FlashInfer MoE 层
# ==========================================
class FlashInferMoELayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(hidden_size, num_experts, top_k)
        
        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_size * 2, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        routing_weights, selected_experts = self.router(x_flat)
        
        result = cutlass_fused_moe(
            input=x_flat,
            token_selected_experts=selected_experts.int().contiguous(),
            token_final_scales=routing_weights.contiguous(),
            fc1_expert_weights=self.w1,
            fc2_expert_weights=self.w2,
            output_dtype=x_flat.dtype,
            quant_scales=None 
        )
        out_flat = result[0] if isinstance(result, (list, tuple)) else result
        return out_flat.view(batch_size, seq_len, hidden_size)


# ==========================================
# 性能压测与对齐脚本
# ==========================================
def run_benchmark(batch_size, seq_len, hidden_size, intermediate_size, num_experts, top_k):
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    total_tokens = batch_size * seq_len
    stage = "Prefill (大批量)" if total_tokens > 512 else "Decode (小批量)"
    
    print(f"\n" + "=" * 60)
    print(f"测试场景: {stage} | {total_tokens} Tokens")
    print(f"配置: Experts={num_experts}, TopK={top_k}, H={hidden_size}, INTER={intermediate_size}")
    print("=" * 60)

    # 1. 初始化模型
    pt_model = PyTorchMoELayer(hidden_size, intermediate_size, num_experts, top_k).to(device).to(dtype)
    fi_model = FlashInferMoELayer(hidden_size, intermediate_size, num_experts, top_k).to(device).to(dtype)

    # 2. 严格的权重同步 (将 PyTorch 随机生成的权重拷贝给 FlashInfer)
    with torch.no_grad():
        fi_model.router.classifier.weight.data.copy_(pt_model.router.classifier.weight.data)
        
        # 将 nn.ModuleList 中的离散权重 stack 起来，直接喂给 FlashInfer 的大 Parameter
        w1_stacked = torch.stack([e.w1.weight.data for e in pt_model.experts])
        w2_stacked = torch.stack([e.w2.weight.data for e in pt_model.experts])
        fi_model.w1.data.copy_(w1_stacked)
        fi_model.w2.data.copy_(w2_stacked)

    # 3. 构造输入 (除以 100 防止 FP16 的连乘导致 NaN)
    x = (torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32) * 0.01).to(dtype)

    # 4. 预热与精度校验
    print("Warming up and checking precision...")
    for _ in range(5):
        out_pt = pt_model(x)
        out_fi = fi_model(x)
    torch.cuda.synchronize()
    
    max_diff = (out_pt - out_fi).abs().max().item()
    print(f"Max Precision Difference: {max_diff:.6f}")

    # 5. 性能打擂台
    iters = 100
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 压测 PyTorch
    start_event.record()
    for _ in range(iters):
        _ = pt_model(x)
    end_event.record()
    torch.cuda.synchronize()
    pt_time = start_event.elapsed_time(end_event) / iters

    # 压测 FlashInfer
    start_event.record()
    for _ in range(iters):
        _ = fi_model(x)
    end_event.record()
    torch.cuda.synchronize()
    fi_time = start_event.elapsed_time(end_event) / iters

    print(f"\nPyTorch Time:    {pt_time:.3f} ms")
    print(f"FlashInfer Time: {fi_time:.3f} ms")
    
    if pt_time > fi_time:
        print(f"🚀 Speedup:        {pt_time / fi_time:.2f}x (FlashInfer 胜)")
    else:
        print(f"🐢 Speedup:        {pt_time / fi_time:.2f}x (PyTorch 胜)")

if __name__ == "__main__":
    H = 4096
    INTER = 14336
    E = 8
    K = 2
    
    # 测试 1: 模拟 Prefill 阶段 (长提示词输入，大量 Token)
    # 预期: FlashInfer 利用 CUTLASS 的连续显存读取，比 for 循环快
    run_benchmark(batch_size=1, seq_len=4096, hidden_size=H, intermediate_size=INTER, num_experts=E, top_k=K)

    # 测试 2: 模拟 Decode 阶段 (自回归逐字生成，小量 Token)
    # 预期: 再次验证我们之前讨论过的 CUTLASS Grouped GEMM 的 CPU 同步开销瓶颈
    run_benchmark(batch_size=32, seq_len=1, hidden_size=H, intermediate_size=INTER, num_experts=E, top_k=K)