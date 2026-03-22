"""
@triton_op 效果对比 Demo
===========================
在 torch.compile 条件下，对比使用 @triton_op 修饰 vs 不使用的效果差异。

核心对比维度:
1. Graph Break 数量    —— @triton_op 避免 graph break，裸调用会断图
2. 推理延迟 (compiled) —— graph break 越少，compile 优化空间越大
3. 训练延迟 (compiled) —— @triton_op + register_autograd 让 fwd+bwd 一体编译
4. 生成的 kernel 数量  —— 通过 compile 日志查看算子融合程度

用法:
    python tritonOp.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

DEVICE = torch.device("cuda:0")


# ============================================================================
# 共享的底层 Triton Kernel (两套 wrapper 共用同一个 kernel)
# ============================================================================

@triton.jit
def _rmsnorm_kernel(
    X, Y, W,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm Triton kernel: y = x * rsqrt(mean(x^2) + eps) * w"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    rms = tl.sum(x * x, axis=0) / N
    inv_rms = 1.0 / tl.sqrt(rms + eps)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    y = x * inv_rms * w
    tl.store(Y + row * stride + cols, y, mask=mask)


@triton.jit
def _fused_gelu_bias_kernel(
    X, BIAS, Y,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused GELU(x + bias) kernel, sigmoid 近似"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(BIAS + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + b
    sigmoid = 1.0 / (1.0 + tl.exp(-1.702 * x))
    y = x * sigmoid
    tl.store(Y + row * N + cols, y, mask=mask)


# ============================================================================
# 方案 A: 使用 @triton_op (推荐方式)
#   - torch.compile 可以 trace 进 Triton kernel
#   - 不会产生 graph break
#   - 支持 register_autograd
# ============================================================================

@triton_op("with_triton_op::rmsnorm", mutates_args={})
def rmsnorm_with_triton_op(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    y = torch.empty_like(x)
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    wrap_triton(_rmsnorm_kernel)[(M,)](x, y, weight, x.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y


@triton_op("with_triton_op::fused_gelu_bias", mutates_args={})
def fused_gelu_bias_with_triton_op(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    wrap_triton(_fused_gelu_bias_kernel)[(M,)](x, bias, y, N, BLOCK_SIZE=BLOCK_SIZE)
    return y


# register_autograd 让 triton_op 支持训练
def _rmsnorm_bwd_with_op(ctx, grad_output):
    x, weight = ctx.saved_tensors
    eps = ctx.eps
    rms = (x.float().pow(2).mean(-1, keepdim=True) + eps).sqrt()
    inv_rms = 1.0 / rms
    x_hat = x.float() * inv_rms
    dy = grad_output.float()
    w = weight.float()
    dweight = (dy * x_hat).sum(dim=0).to(weight.dtype)
    dx = w * inv_rms * (dy - x_hat * (dy * x_hat).mean(dim=-1, keepdim=True))
    return dx.to(x.dtype), dweight

def _rmsnorm_setup_with_op(ctx, inputs, output):
    x, weight, eps = inputs
    ctx.save_for_backward(x, weight)
    ctx.eps = eps

rmsnorm_with_triton_op.register_autograd(_rmsnorm_bwd_with_op, setup_context=_rmsnorm_setup_with_op)


def _gelu_bias_bwd_with_op(ctx, grad_output):
    x, bias = ctx.saved_tensors
    z = x.float() + bias.float()
    s = torch.sigmoid(1.702 * z)
    grad_act = s + 1.702 * z * s * (1.0 - s)
    dx = (grad_output.float() * grad_act).to(x.dtype)
    dbias = dx.sum(dim=0).to(bias.dtype)
    return dx, dbias

def _gelu_bias_setup_with_op(ctx, inputs, output):
    x, bias = inputs
    ctx.save_for_backward(x, bias)

fused_gelu_bias_with_triton_op.register_autograd(_gelu_bias_bwd_with_op, setup_context=_gelu_bias_setup_with_op)


# ============================================================================
# 方案 B: 不使用 @triton_op (裸调用 Triton kernel)
#   - torch.compile 无法 trace，视为不透明的 Python 函数
#   - 产生 graph break
#   - 不支持 compile 图内优化
# ============================================================================

def rmsnorm_without_triton_op(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """裸调用 Triton kernel，没有 @triton_op 修饰"""
    y = torch.empty_like(x)
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    _rmsnorm_kernel[(M,)](x, y, weight, x.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y


def fused_gelu_bias_without_triton_op(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """裸调用 Triton kernel，没有 @triton_op 修饰"""
    y = torch.empty_like(x)
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    _fused_gelu_bias_kernel[(M,)](x, bias, y, N, BLOCK_SIZE=BLOCK_SIZE)
    return y


# ============================================================================
# 用两套 wrapper 构建同一个模型结构
# ============================================================================

class RMSNormModule(nn.Module):
    def __init__(self, hidden_size, use_triton_op=True, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.use_triton_op = use_triton_op

    def forward(self, x):
        orig_shape = x.shape
        x_2d = x.view(-1, orig_shape[-1]).contiguous()
        if self.use_triton_op:
            y = rmsnorm_with_triton_op(x_2d, self.weight, self.eps)
        else:
            y = rmsnorm_without_triton_op(x_2d, self.weight, self.eps)
        return y.view(orig_shape)


class FusedGeluMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, use_triton_op=True):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.use_triton_op = use_triton_op

    def forward(self, x):
        h = F.linear(x, self.up_proj.weight)
        orig_shape = h.shape
        h_2d = h.view(-1, orig_shape[-1]).contiguous()
        if self.use_triton_op:
            h_2d = fused_gelu_bias_with_triton_op(h_2d, self.up_proj.bias)
        else:
            h_2d = fused_gelu_bias_without_triton_op(h_2d, self.up_proj.bias)
        return self.down_proj(h_2d.view(orig_shape))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, use_triton_op=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_norm = RMSNormModule(hidden_size, use_triton_op)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_norm = RMSNormModule(hidden_size, use_triton_op)
        self.mlp = FusedGeluMLP(hidden_size, intermediate_size, use_triton_op)

    def forward(self, x):
        B, S, D = x.shape
        residual = x
        x = self.attn_norm(x)
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        x = residual + self.o_proj(attn_out)
        residual = x
        x = self.mlp_norm(x)
        x = residual + self.mlp(x)
        return x


class DemoModel(nn.Module):
    def __init__(self, hidden_size=256, num_heads=4, num_layers=2,
                 intermediate_size=512, use_triton_op=True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size, use_triton_op)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNormModule(hidden_size, use_triton_op)
        self.head = nn.Linear(hidden_size, 32)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.head(x[:, 0, :])


# ============================================================================
# Benchmark 工具
# ============================================================================

def benchmark_latency(fn, warmup=50, repeat=200):
    """CUDA event 测量延迟 (ms), 返回 P50"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
    return times[len(times) // 2]


# ============================================================================
# 测试 1: Graph Break 对比
# ============================================================================

def test_graph_breaks():
    print("=" * 70)
    print("测试 1: Graph Break 对比  (torch.compile 下)")
    print("=" * 70)
    print()
    print("  说明: @triton_op + wrap_triton 让 compile 能 trace 进 Triton kernel,")
    print("  裸调用 Triton kernel 是不透明的 Python 函数，compile 必须在此处断图。")
    print()

    B, S, D = 2, 16, 256
    x = torch.randn(B, S, D, device=DEVICE)

    for name, use_triton_op in [("有 @triton_op", True), ("无 @triton_op (裸调用)", False)]:
        model = DemoModel(use_triton_op=use_triton_op).to(DEVICE).eval()
        torch._dynamo.reset()
        try:
            explanation = torch._dynamo.explain(model)(x)
            print(f"  [{name}]")
            print(f"    Graph 数量:       {explanation.graph_count}")
            print(f"    Graph Break 数量: {explanation.graph_break_count}")
            if explanation.break_reasons:
                for i, reason in enumerate(explanation.break_reasons[:3]):  # 最多显示3条
                    print(f"      Break {i+1}: {reason}")
                if len(explanation.break_reasons) > 3:
                    print(f"      ... 还有 {len(explanation.break_reasons) - 3} 个 break")
            print()
        except Exception as e:
            print(f"  [{name}] explain 失败: {e}")
            print()

    torch._dynamo.reset()


# ============================================================================
# 测试 2: 推理性能对比 (都开启 torch.compile)
# ============================================================================

def test_inference_compiled():
    print("=" * 70)
    print("测试 2: 推理性能对比  (都开启 torch.compile)")
    print("=" * 70)
    print()

    B, S, D = 8, 128, 256
    x = torch.randn(B, S, D, device=DEVICE)

    results = {}

    with torch.no_grad():
        for name, use_triton_op, mode in [
            ("有 @triton_op + compile(default)",          True,  "default"),
            ("无 @triton_op + compile(default)",          False, "default"),
            ("有 @triton_op + compile(reduce-overhead)",  True,  "reduce-overhead"),
            ("无 @triton_op + compile(reduce-overhead)",  False, "reduce-overhead"),
            ("有 @triton_op + compile(max-autotune)",     True,  "max-autotune"),
            ("无 @triton_op + compile(max-autotune)",     False, "max-autotune"),
        ]:
            torch._dynamo.reset()
            model = DemoModel(use_triton_op=use_triton_op).to(DEVICE).eval()
            compiled_model = torch.compile(model, mode=mode)
            latency = benchmark_latency(lambda: compiled_model(x), warmup=30, repeat=150)
            results[name] = latency
            print(f"  {name:<48s}  {latency:>8.3f} ms")

    # 分组对比
    print()
    print("  ── 同 mode 下的加速对比 ──")
    for mode in ["default", "reduce-overhead", "max-autotune"]:
        key_with = f"有 @triton_op + compile({mode})"
        key_without = f"无 @triton_op + compile({mode})"
        lat_with = results[key_with]
        lat_without = results[key_without]
        ratio = lat_without / lat_with
        winner = "@triton_op 更快" if ratio > 1.02 else ("相当" if ratio > 0.98 else "裸调用更快")
        print(f"    {mode:<20s}  有op={lat_with:.3f}ms  无op={lat_without:.3f}ms  比值={ratio:.2f}x  → {winner}")

    print()
    torch._dynamo.reset()


# ============================================================================
# 测试 3: 训练性能对比 (都开启 torch.compile)
# ============================================================================

def test_training_compiled():
    print("=" * 70)
    print("测试 3: 训练性能对比  (都开启 torch.compile)")
    print("=" * 70)
    print()
    print("  说明: 没有 @triton_op 时，裸 Triton kernel 不支持 autograd，")
    print("  训练会直接报错。这里对比两种可训练方案:")
    print("    A) @triton_op + register_autograd + compile  (Triton 方式)")
    print("    B) 纯 PyTorch 算子 + compile                 (对照组)")
    print()

    B, S, D = 8, 128, 256
    num_classes = 32
    x = torch.randn(B, S, D, device=DEVICE)
    target = torch.randint(0, num_classes, (B,), device=DEVICE)

    def train_step(model, optimizer, x, target):
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        return loss

    results = {}

    # ---- A) 有 @triton_op + compile 训练 ----
    torch._dynamo.reset()
    model_a = DemoModel(use_triton_op=True).to(DEVICE).train()
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-4)
    compiled_step_a = torch.compile(train_step, mode="reduce-overhead")
    latency = benchmark_latency(lambda: compiled_step_a(model_a, opt_a, x, target), warmup=30, repeat=100)
    results["@triton_op + compile 训练"] = latency
    print(f"  [A] @triton_op + register_autograd + compile:  {latency:.3f} ms/step")

    # ---- B) 无 @triton_op 训练: 尝试裸调用, 预期会失败 ----
    print()
    print("  [B] 裸 Triton kernel (无 @triton_op) + compile 训练:")
    torch._dynamo.reset()
    model_b = DemoModel(use_triton_op=False).to(DEVICE).train()
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-4)
    try:
        compiled_step_b = torch.compile(train_step, mode="reduce-overhead")
        # warmup 阶段就会报错
        for _ in range(5):
            compiled_step_b(model_b, opt_b, x, target)
        # 如果没报错（某些版本可能 fallback），测量性能
        latency = benchmark_latency(lambda: compiled_step_b(model_b, opt_b, x, target), warmup=10, repeat=50)
        results["裸调用 + compile 训练"] = latency
        print(f"      竟然没报错! 延迟: {latency:.3f} ms/step (可能 dynamo fallback 到 eager)")
    except Exception as e:
        error_msg = str(e).split('\n')[0][:100]
        print(f"      ❌ 报错: {error_msg}")
        print(f"      原因: 裸 Triton kernel 没有注册 autograd，backward() 无法计算梯度")
        results["裸调用 + compile 训练"] = float('inf')

    # ---- C) 纯 PyTorch 对照组 + compile 训练 ----
    torch._dynamo.reset()

    class PyTorchModel(nn.Module):
        """纯 PyTorch 实现, 等价的 RMSNorm + GELU 结构"""
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([self._make_block() for _ in range(2)])
            self.final_norm = nn.LayerNorm(256)
            self.head = nn.Linear(256, 32)

        def _make_block(self):
            return nn.ModuleDict({
                'attn_norm': nn.LayerNorm(256),
                'q_proj': nn.Linear(256, 256, bias=False),
                'k_proj': nn.Linear(256, 256, bias=False),
                'v_proj': nn.Linear(256, 256, bias=False),
                'o_proj': nn.Linear(256, 256, bias=False),
                'mlp_norm': nn.LayerNorm(256),
                'up_proj': nn.Linear(256, 512),
                'act': nn.GELU(),
                'down_proj': nn.Linear(512, 256),
            })

        def forward(self, x):
            B, S, D = x.shape
            for block in self.layers:
                residual = x
                x = block['attn_norm'](x)
                q = block['q_proj'](x).view(B, S, 4, 64).transpose(1, 2)
                k = block['k_proj'](x).view(B, S, 4, 64).transpose(1, 2)
                v = block['v_proj'](x).view(B, S, 4, 64).transpose(1, 2)
                attn_out = F.scaled_dot_product_attention(q, k, v)
                attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
                x = residual + block['o_proj'](attn_out)
                residual = x
                x = block['mlp_norm'](x)
                x = residual + block['down_proj'](block['act'](block['up_proj'](x)))
            x = self.final_norm(x)
            return self.head(x[:, 0, :])

    model_c = PyTorchModel().to(DEVICE).train()
    opt_c = torch.optim.AdamW(model_c.parameters(), lr=1e-4)
    compiled_step_c = torch.compile(train_step, mode="reduce-overhead")
    latency = benchmark_latency(lambda: compiled_step_c(model_c, opt_c, x, target), warmup=30, repeat=100)
    results["纯 PyTorch + compile 训练"] = latency
    print(f"  [C] 纯 PyTorch + compile 训练:                  {latency:.3f} ms/step")

    # 汇总
    print()
    print("  ── 训练性能汇总 ──")
    for name, lat in results.items():
        if lat == float('inf'):
            print(f"    {name:<35s}    ❌ 无法训练")
        else:
            print(f"    {name:<35s}  {lat:>8.3f} ms/step")
    print()
    torch._dynamo.reset()


# ============================================================================
# 测试 4: 单算子级别对比 (compile 下)
# ============================================================================

def test_single_operator():
    print("=" * 70)
    print("测试 4: 单算子级别对比  (torch.compile 下)")
    print("=" * 70)
    print()
    print("  对比同一个 Triton kernel，有/无 @triton_op，经过 compile 后的差异。")
    print()

    M, N = 4096, 1024
    x = torch.randn(M, N, device=DEVICE)
    w = torch.ones(N, device=DEVICE)

    with torch.no_grad():
        # ---- RMSNorm ----
        print("  >> RMSNorm (M=4096, N=1024)")

        # 有 @triton_op + compile
        torch._dynamo.reset()
        compiled_with = torch.compile(rmsnorm_with_triton_op)
        lat_with = benchmark_latency(lambda: compiled_with(x, w))

        # 无 @triton_op + compile
        torch._dynamo.reset()
        compiled_without = torch.compile(rmsnorm_without_triton_op)
        lat_without = benchmark_latency(lambda: compiled_without(x, w))

        print(f"     有 @triton_op + compile:  {lat_with:.3f} ms")
        print(f"     无 @triton_op + compile:  {lat_without:.3f} ms")
        ratio = lat_without / lat_with
        print(f"     比值: {ratio:.2f}x")

        # ---- Fused GELU + Bias ----
        print()
        print("  >> Fused GELU+Bias (M=4096, N=1024)")
        bias = torch.randn(N, device=DEVICE)

        torch._dynamo.reset()
        compiled_with = torch.compile(fused_gelu_bias_with_triton_op)
        lat_with = benchmark_latency(lambda: compiled_with(x, bias))

        torch._dynamo.reset()
        compiled_without = torch.compile(fused_gelu_bias_without_triton_op)
        lat_without = benchmark_latency(lambda: compiled_without(x, bias))

        print(f"     有 @triton_op + compile:  {lat_with:.3f} ms")
        print(f"     无 @triton_op + compile:  {lat_without:.3f} ms")
        ratio = lat_without / lat_with
        print(f"     比值: {ratio:.2f}x")

    print()
    torch._dynamo.reset()


# ============================================================================
# 测试 5: compile 日志分析 —— 查看算子融合情况
# ============================================================================

def test_compile_logs():
    print("=" * 70)
    print("测试 5: torch.compile 编译行为分析")
    print("=" * 70)
    print()
    print("  使用 torch._dynamo.explain 深入分析两种方案的编译差异。")
    print()

    B, S, D = 2, 16, 256
    x = torch.randn(B, S, D, device=DEVICE)

    for name, use_triton_op in [("有 @triton_op", True), ("无 @triton_op", False)]:
        model = DemoModel(use_triton_op=use_triton_op).to(DEVICE).eval()
        torch._dynamo.reset()

        print(f"  [{name}]")
        try:
            explanation = torch._dynamo.explain(model)(x)
            print(f"    Graph 数量:       {explanation.graph_count}")
            print(f"    Graph Break 数量: {explanation.graph_break_count}")
            print(f"    Ops per graph:    ~{explanation.ops_per_graph}")

            # 分析含义
            if explanation.graph_break_count == 0:
                print(f"    ✅ 整个模型编译为单一计算图，compile 可以做全局优化:")
                print(f"       - 算子融合 (相邻 elementwise ops 合并)")
                print(f"       - 内存规划 (减少中间 tensor 分配)")
                print(f"       - CUDA Graph 捕获 (消除 kernel launch overhead)")
            else:
                n_breaks = explanation.graph_break_count
                n_graphs = explanation.graph_count
                print(f"    ⚠️  模型被切成 {n_graphs} 个子图，每个子图独立编译:")
                print(f"       - 子图之间无法做算子融合")
                print(f"       - 每个断点处需要同步 + Python 回退")
                print(f"       - CUDA Graph 只能分段捕获 (或无法捕获)")
                print(f"       - 实际 kernel launch 次数 ≈ 正常的 {n_graphs}x")
        except Exception as e:
            print(f"    explain 失败: {e}")
        print()

    torch._dynamo.reset()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print()
    print("🔬 @triton_op 效果对比: 有 vs 无, 都开 torch.compile")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   Triton:  {triton.__version__}")
    print(f"   GPU:     {torch.cuda.get_device_name(0)}")
    print()

    # 1. Graph Break 是核心差异
    test_graph_breaks()

    # 2. 推理性能
    test_inference_compiled()

    # 3. 训练性能 (register_autograd 的价值)
    test_training_compiled()

    # 4. 单算子对比
    test_single_operator()

    # 5. 编译行为分析
    test_compile_logs()

    # 总结
    print("=" * 70)
    print("📋 结论总结")
    print("=" * 70)
    print("""
  ┌─────────────────────────┬──────────────────────┬──────────────────────┐
  │                         │  有 @triton_op       │  无 @triton_op       │
  │                         │  + wrap_triton       │  (裸调用 kernel)     │
  ├─────────────────────────┼──────────────────────┼──────────────────────┤
  │ compile 能否 trace      │  ✅ 能               │  ❌ 不能             │
  │ Graph Break             │  ✅ 无               │  ⚠️  每个 kernel 断一次│
  │ 算子融合                │  ✅ 与前后 ops 融合   │  ❌ 无法融合         │
  │ CUDA Graph 捕获         │  ✅ 整图捕获          │  ⚠️  分段或失败      │
  │ register_autograd       │  ✅ 支持              │  ❌ 不支持           │
  │ 训练 backward           │  ✅ 可训练            │  ❌ 报错             │
  │ compile + 训练          │  ✅ fwd+bwd 一体编译  │  ❌ 无法工作         │
  └─────────────────────────┴──────────────────────┴──────────────────────┘

  核心原理:
    @triton_op    → 把 Triton kernel 注册为 PyTorch 算子 (torch.ops.xxx)
    wrap_triton() → 告诉 Inductor 这是 Triton kernel，可以纳入计算图
    两者配合      → compile trace 时不断图，Inductor 可以联合优化

  何时 @triton_op 优势最大:
    1. 模型中有多个自定义 Triton kernel → 避免反复 graph break
    2. 需要训练 (backward) → register_autograd 是唯一正确方式
    3. 使用 reduce-overhead / CUDA Graph → 需要单一计算图
""")
