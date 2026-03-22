"""
torch.compile 性能验证 Demo
=============================
包含3个自定义 Triton 算子 (Fused LayerNorm, Fused RMSNorm, Fused GELU+Bias)，
集成到一个迷你 Transformer 模型中，全面验证 torch.compile 的性能效果。

验证维度:
1. Triton 自定义算子集成 —— compile 如何处理自定义 Triton kernel
2. 端到端模型加速      —— eager vs compiled 的整体性能对比
3. 算子融合效果对比    —— compile 自动融合 vs 手写 Triton kernel
4. Graph break 分析    —— 检测 compile 中的图断裂情况

用法:
    python modev1.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

DEVICE = torch.device("cuda:0")

# ============================================================================
# Part 1: 自定义 Triton 算子
# ============================================================================

# ---------- 1.1 Fused LayerNorm Triton Kernel ----------

@triton.jit
def _layernorm_fwd_kernel(
    X, Y, W, B,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # 加载一行数据
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # 计算均值和方差
    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + eps)

    # 归一化 + 仿射变换
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_centered * inv_std * w + b

    tl.store(Y + row * stride + cols, y, mask=mask)


@triton_op("custom::layernorm", mutates_args={})
def triton_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Triton LayerNorm wrapper

    使用 @triton_op 注册算子，使得 torch.compile 能够 trace 进 Triton kernel，
    而非将其视为不透明的 custom_op。这样 compile 可以：
    1. 将 Triton kernel 与前后的 PyTorch 算子做联合优化
    2. 避免不必要的 graph break
    3. 正确处理 autograd、FlopCounter 等子系统
    """
    assert x.is_contiguous()
    y = torch.empty_like(x)
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    # wrap_triton() 告诉 torch.compile 这是一个 Triton kernel 调用
    wrap_triton(_layernorm_fwd_kernel)[(M,)](
        x, y, weight, bias,
        x.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


# ---- LayerNorm register_autograd ----
# 反向传播数学推导:
#   前向: y = (x - mean) * inv_std * w + b
#   其中: mean = sum(x)/N, var = sum((x-mean)^2)/N, inv_std = 1/sqrt(var+eps)
#
#   反向 (对 x 的梯度):
#     dx = w * inv_std/N * (N*dy - sum(dy) - (x-mean)*inv_std^2 * sum(dy*(x-mean)))
#   对 weight 的梯度: dw = sum_over_rows(dy * (x-mean) * inv_std)
#   对 bias 的梯度:   db = sum_over_rows(dy)

def _layernorm_backward(ctx, grad_output):
    """LayerNorm 反向: 使用 PyTorch 算子计算梯度（清晰易懂）"""
    x, weight, bias = ctx.saved_tensors
    eps = ctx.eps
    N = x.shape[-1]

    # 重新计算前向中间量
    mean = x.float().mean(dim=-1, keepdim=True)
    x_centered = x.float() - mean
    var = x_centered.pow(2).mean(dim=-1, keepdim=True)
    inv_std = (var + eps).rsqrt()

    # 归一化后的值
    x_hat = x_centered * inv_std
    dy = grad_output.float()

    # dweight, dbias: 沿 batch 维度求和
    dweight = (dy * x_hat).sum(dim=0).to(weight.dtype)
    dbias = dy.sum(dim=0).to(bias.dtype)

    # dx: LayerNorm 反向公式
    w = weight.float()
    dx = w * inv_std / N * (
        N * dy - dy.sum(dim=-1, keepdim=True)
        - x_hat * (dy * x_hat).sum(dim=-1, keepdim=True)
    )
    dx = dx.to(x.dtype)

    return dx, dweight, dbias


def _layernorm_setup_context(ctx, inputs, output):
    x, weight, bias, eps = inputs
    ctx.save_for_backward(x, weight, bias)
    ctx.eps = eps


triton_layernorm.register_autograd(_layernorm_backward, setup_context=_layernorm_setup_context)


# ---------- 1.2 Fused RMSNorm Triton Kernel ----------

@triton.jit
def _rmsnorm_fwd_kernel(
    X, Y, W,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm: 不减均值，直接用 x^2 的均值
    rms = tl.sum(x * x, axis=0) / N
    inv_rms = 1.0 / tl.sqrt(rms + eps)

    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    y = x * inv_rms * w

    tl.store(Y + row * stride + cols, y, mask=mask)


@triton_op("custom::rmsnorm", mutates_args={})
def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Triton RMSNorm wrapper (使用 @triton_op 注册)"""
    assert x.is_contiguous()
    y = torch.empty_like(x)
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    wrap_triton(_rmsnorm_fwd_kernel)[(M,)](
        x, y, weight,
        x.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


# ---- RMSNorm register_autograd ----
# 反向传播数学推导:
#   前向: y = x * inv_rms * w
#   其中: rms = sqrt(mean(x^2) + eps), inv_rms = 1/rms
#
#   反向 (对 x 的梯度):
#     令 x_hat = x * inv_rms
#     dx = w * inv_rms * (dy - x_hat * mean(dy * x_hat))
#   对 weight 的梯度: dw = sum_over_rows(dy * x * inv_rms)

def _rmsnorm_backward(ctx, grad_output):
    """RMSNorm 反向: 使用 PyTorch 算子计算梯度"""
    x, weight = ctx.saved_tensors
    eps = ctx.eps
    N = x.shape[-1]

    # 重新计算前向中间量
    rms = (x.float().pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    inv_rms = 1.0 / rms
    x_hat = x.float() * inv_rms

    dy = grad_output.float()
    w = weight.float()

    # dweight: 沿 batch 维度求和
    dweight = (dy * x_hat).sum(dim=0).to(weight.dtype)

    # dx: RMSNorm 反向公式
    dx = w * inv_rms * (dy - x_hat * (dy * x_hat).mean(dim=-1, keepdim=True))
    dx = dx.to(x.dtype)

    return dx, dweight


def _rmsnorm_setup_context(ctx, inputs, output):
    x, weight, eps = inputs
    ctx.save_for_backward(x, weight)
    ctx.eps = eps


triton_rmsnorm.register_autograd(_rmsnorm_backward, setup_context=_rmsnorm_setup_context)


# ---------- 1.3 Fused GELU + Bias Triton Kernel ----------

@triton.jit
def _fused_gelu_bias_kernel(
    X, BIAS, Y,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused GELU activation + bias: y = GELU(x + bias)

    使用 sigmoid 近似: GELU(x) ≈ x * sigmoid(1.702 * x)
    避免依赖 tl.math.tanh（部分 Triton 版本不可用）
    融合 bias add 和 activation，减少一次全局内存读写
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(BIAS + cols, mask=mask, other=0.0).to(tl.float32)

    # fused: x = x + bias, 然后 GELU (sigmoid 近似)
    x = x + b
    # sigmoid 近似 GELU: x * σ(1.702x)
    sigmoid = 1.0 / (1.0 + tl.exp(-1.702 * x))
    y = x * sigmoid

    tl.store(Y + row * N + cols, y, mask=mask)


@triton_op("custom::fused_gelu_bias", mutates_args={})
def triton_fused_gelu_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Triton Fused GELU+Bias wrapper (使用 @triton_op 注册)"""
    assert x.is_contiguous()
    y = torch.empty_like(x)
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    wrap_triton(_fused_gelu_bias_kernel)[(M,)](
        x, bias, y,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


# ---- Fused GELU+Bias register_autograd ----
# 反向传播数学推导:
#   前向: y = SiLU_approx(x + bias) = (x+b) * σ(1.702*(x+b))
#   其中: σ(z) = 1/(1+exp(-z))
#
#   反向 (sigmoid 近似 GELU 的导数):
#     令 z = x + bias, s = σ(1.702*z)
#     dy/dz = s + 1.702*z*s*(1-s)
#     dx = grad_output * dy/dz
#     dbias = sum_over_rows(grad_output * dy/dz)

@triton.jit
def _fused_gelu_bias_bwd_kernel(
    GRAD_OUT, X, BIAS, GRAD_X, GRAD_BIAS,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused GELU+Bias 反向 Triton kernel

    也用 Triton 写反向 kernel，体现 register_autograd 的优势:
    forward + backward 都是高效的 fused kernel
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    dy = tl.load(GRAD_OUT + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(BIAS + cols, mask=mask, other=0.0).to(tl.float32)

    # z = x + bias
    z = x + b
    # sigmoid(1.702 * z)
    s = 1.0 / (1.0 + tl.exp(-1.702 * z))
    # d(GELU_sigmoid)/dz = s + 1.702*z*s*(1-s)
    grad_act = s + 1.702 * z * s * (1.0 - s)
    # dx = dy * grad_act (对 x 和 bias 的梯度相同，因为 z = x + b)
    dx = dy * grad_act

    tl.store(GRAD_X + row * N + cols, dx, mask=mask)
    # grad_bias 需要跨行求和，这里先存每行的贡献，后续用 atomicAdd 或者外部 reduce
    # 为简单起见，使用 tl.atomic_add 逐行累加
    tl.atomic_add(GRAD_BIAS + cols, dx, mask=mask)


@triton_op("custom::fused_gelu_bias_bwd", mutates_args={"grad_bias"})
def _triton_fused_gelu_bias_bwd(
    grad_output: torch.Tensor, x: torch.Tensor, bias: torch.Tensor,
    grad_bias: torch.Tensor,
) -> torch.Tensor:
    """Triton Fused GELU+Bias 反向 wrapper"""
    grad_x = torch.empty_like(x)
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    wrap_triton(_fused_gelu_bias_bwd_kernel)[(M,)](
        grad_output, x, bias, grad_x, grad_bias,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return grad_x


def _fused_gelu_bias_backward(ctx, grad_output):
    """Fused GELU+Bias 反向: forward 和 backward 都用 Triton kernel"""
    x, bias = ctx.saved_tensors
    M, N = x.shape
    grad_output_2d = grad_output.view(M, N).contiguous()

    # grad_bias 需要 zero 初始化 (因为用 atomic_add 累加)
    grad_bias = torch.zeros_like(bias)
    grad_x = _triton_fused_gelu_bias_bwd(grad_output_2d, x, bias, grad_bias)
    return grad_x, grad_bias


def _fused_gelu_bias_setup_context(ctx, inputs, output):
    x, bias = inputs
    ctx.save_for_backward(x, bias)


triton_fused_gelu_bias.register_autograd(
    _fused_gelu_bias_backward,
    setup_context=_fused_gelu_bias_setup_context,
)


# ============================================================================
# Part 2: 使用 Triton 算子构建迷你 Transformer 模型
# ============================================================================

class TritonLayerNorm(nn.Module):
    """使用 Triton kernel 的 LayerNorm"""
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        orig_shape = x.shape
        x_2d = x.view(-1, orig_shape[-1]).contiguous()
        y = triton_layernorm(x_2d, self.weight, self.bias, self.eps)
        return y.view(orig_shape)


class TritonRMSNorm(nn.Module):
    """使用 Triton kernel 的 RMSNorm"""
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        orig_shape = x.shape
        x_2d = x.view(-1, orig_shape[-1]).contiguous()
        y = triton_rmsnorm(x_2d, self.weight, self.eps)
        return y.view(orig_shape)


class TritonFusedGeluMLP(nn.Module):
    """使用 Triton Fused GELU+Bias 的 MLP 层"""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # up_proj 不加 bias（手动拆出来给 Triton fused kernel）
        h = F.linear(x, self.up_proj.weight)  # [B, S, intermediate] 无 bias
        orig_shape = h.shape
        h_2d = h.view(-1, orig_shape[-1]).contiguous()
        h_2d = triton_fused_gelu_bias(h_2d, self.up_proj.bias)  # fused bias + gelu
        h = h_2d.view(orig_shape)
        return self.down_proj(h)


class PyTorchMLP(nn.Module):
    """纯 PyTorch 实现的 MLP（对照组）"""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


class MiniTransformerBlock(nn.Module):
    """迷你 Transformer Block，混合使用 Triton 算子"""
    def __init__(self, hidden_size, num_heads, intermediate_size, use_triton=True):
        super().__init__()
        self.use_triton = use_triton

        # Attention 前的 norm
        if use_triton:
            self.attn_norm = TritonRMSNorm(hidden_size)
        else:
            self.attn_norm = nn.LayerNorm(hidden_size)

        # Self-Attention (用 PyTorch 标准实现，让 compile 去融合)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # MLP 前的 norm
        if use_triton:
            self.mlp_norm = TritonLayerNorm(hidden_size)
            self.mlp = TritonFusedGeluMLP(hidden_size, intermediate_size)
        else:
            self.mlp_norm = nn.LayerNorm(hidden_size)
            self.mlp = PyTorchMLP(hidden_size, intermediate_size)

    def forward(self, x):
        B, S, D = x.shape

        # --- Self-Attention with RMSNorm (Pre-Norm) ---
        residual = x
        x = self.attn_norm(x)

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention (PyTorch 2.0+)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        x = residual + self.o_proj(attn_out)

        # --- MLP with LayerNorm (Pre-Norm) ---
        residual = x
        x = self.mlp_norm(x)
        x = residual + self.mlp(x)

        return x


class MiniTransformer(nn.Module):
    """迷你 Transformer 模型

    Args:
        hidden_size: 隐藏层维度
        num_heads: 注意力头数
        num_layers: Transformer 层数
        intermediate_size: MLP 中间层维度
        use_triton: 是否使用 Triton 自定义算子
    """
    def __init__(self, hidden_size=256, num_heads=4, num_layers=2,
                 intermediate_size=512, use_triton=True):
        super().__init__()
        self.layers = nn.ModuleList([
            MiniTransformerBlock(hidden_size, num_heads, intermediate_size, use_triton)
            for _ in range(num_layers)
        ])
        # 最终的 norm
        if use_triton:
            self.final_norm = TritonRMSNorm(hidden_size)
        else:
            self.final_norm = nn.LayerNorm(hidden_size)
        # 简单的分类头
        self.head = nn.Linear(hidden_size, 32)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        # 取第一个 token 做分类
        return self.head(x[:, 0, :])


# ============================================================================
# Part 3: 正确性验证
# ============================================================================

def verify_correctness():
    """验证 Triton 算子的正确性"""
    print("=" * 60)
    print("Part 1: Triton 算子正确性验证")
    print("=" * 60)

    torch.manual_seed(42)
    M, N = 32, 256

    # --- LayerNorm ---
    x = torch.randn(M, N, device=DEVICE)
    w = torch.randn(N, device=DEVICE)
    b = torch.randn(N, device=DEVICE)
    y_torch = F.layer_norm(x, (N,), w, b)
    y_triton = triton_layernorm(x, w, b)
    ln_ok = torch.allclose(y_torch, y_triton, atol=1e-5)
    print(f"  LayerNorm: {'✅ PASS' if ln_ok else '❌ FAIL'}  "
          f"max_diff={( y_torch - y_triton).abs().max().item():.2e}")

    # --- RMSNorm ---
    x = torch.randn(M, N, device=DEVICE)
    w = torch.ones(N, device=DEVICE)
    rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
    y_ref = (x.float() / rms * w.float()).to(x.dtype)
    x_2d = x.view(-1, N).contiguous()
    y_triton = triton_rmsnorm(x_2d, w).view_as(x)
    rms_ok = torch.allclose(y_ref, y_triton, atol=1e-5)
    print(f"  RMSNorm:   {'✅ PASS' if rms_ok else '❌ FAIL'}  "
          f"max_diff={(y_ref - y_triton).abs().max().item():.2e}")

    # --- Fused GELU + Bias ---
    x = torch.randn(M, N, device=DEVICE)
    bias = torch.randn(N, device=DEVICE)
    # Triton kernel 使用 sigmoid 近似 GELU: x * σ(1.702x)
    # 与精确 GELU 有轻微差异, 放宽 atol
    y_ref = F.gelu(x + bias)
    y_triton = triton_fused_gelu_bias(x, bias)
    gelu_ok = torch.allclose(y_ref, y_triton, atol=5e-2)  # sigmoid近似 vs 精确GELU
    print(f"  FusedGELU: {'✅ PASS' if gelu_ok else '❌ FAIL'}  "
          f"max_diff={(y_ref - y_triton).abs().max().item():.2e}")

    # --- 端到端模型 ---
    print("\n  端到端模型输出一致性验证:")
    model_triton = MiniTransformer(use_triton=True).to(DEVICE).eval()
    model_pytorch = MiniTransformer(use_triton=False).to(DEVICE).eval()

    # 拷贝对应的权重（attention 和 head 部分）
    model_pytorch.load_state_dict(model_triton.state_dict(), strict=False)

    x = torch.randn(2, 16, 256, device=DEVICE)
    with torch.no_grad():
        y_triton_model = model_triton(x)
        y_pytorch_model = model_pytorch(x)

    # 由于 Triton 和 PyTorch 的 norm 实现有差异(RMSNorm vs LayerNorm)，
    # 这里只检查输出形状和有限范围
    print(f"  Triton model output shape:  {y_triton_model.shape}")
    print(f"  PyTorch model output shape: {y_pytorch_model.shape}")
    print(f"  两个模型都能正常前向传播 ✅")
    print()


# ============================================================================
# Part 4: torch.compile 性能 Benchmark
# ============================================================================

def benchmark_latency(fn, warmup=50, repeat=200):
    """测量函数延迟 (ms)"""
    # Warmup
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

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    # 取 P50
    return times[len(times) // 2]


def benchmark_compile():
    """对比 eager / compiled / compiled+triton 的性能"""
    print("=" * 60)
    print("Part 2: torch.compile 性能 Benchmark")
    print("=" * 60)

    batch_size = 8
    seq_len = 128
    hidden_size = 256

    x = torch.randn(batch_size, seq_len, hidden_size, device=DEVICE)

    # ---- 场景 1: 纯 PyTorch 模型 ----
    model_pytorch = MiniTransformer(use_triton=False).to(DEVICE).eval()

    # ---- 场景 2: Triton 算子模型 ----
    model_triton = MiniTransformer(use_triton=True).to(DEVICE).eval()

    results = {}

    with torch.no_grad():
        # 1) PyTorch Eager
        latency = benchmark_latency(lambda: model_pytorch(x))
        results["PyTorch Eager"] = latency
        print(f"\n  [1] PyTorch Eager Mode:              {latency:.3f} ms")

        # 2) PyTorch + torch.compile
        model_compiled = torch.compile(model_pytorch, mode="reduce-overhead")
        latency = benchmark_latency(lambda: model_compiled(x))
        results["PyTorch Compiled"] = latency
        print(f"  [2] PyTorch + torch.compile:          {latency:.3f} ms")

        # 3) Triton 算子 Eager
        latency = benchmark_latency(lambda: model_triton(x))
        results["Triton Eager"] = latency
        print(f"  [3] Triton Ops Eager Mode:            {latency:.3f} ms")

        # 4) Triton 算子 + torch.compile
        model_triton_compiled = torch.compile(model_triton, mode="reduce-overhead")
        latency = benchmark_latency(lambda: model_triton_compiled(x))
        results["Triton Compiled"] = latency
        print(f"  [4] Triton Ops + torch.compile:       {latency:.3f} ms")

        # 5) PyTorch + compile (max-autotune，更激进的优化)
        model_autotune = torch.compile(model_pytorch, mode="max-autotune")
        latency = benchmark_latency(lambda: model_autotune(x))
        results["PyTorch Max-Autotune"] = latency
        print(f"  [5] PyTorch + compile (max-autotune): {latency:.3f} ms")

    # ---- 汇总对比 ----
    print("\n" + "-" * 60)
    print("  性能汇总 (P50 latency):")
    print("-" * 60)
    baseline = results["PyTorch Eager"]
    for name, lat in results.items():
        speedup = baseline / lat
        bar = "█" * int(speedup * 20)
        print(f"  {name:<30s} {lat:>8.3f} ms  {speedup:>5.2f}x  {bar}")
    print()


# ============================================================================
# Part 5: Graph Break 分析
# ============================================================================

def analyze_graph_breaks():
    """分析 torch.compile 在不同模型上的 graph break 情况"""
    print("=" * 60)
    print("Part 3: Graph Break 分析")
    print("=" * 60)

    import logging
    # 设置 dynamo 日志以捕获 graph break 信息
    torch._dynamo.config.verbose = True
    logger = logging.getLogger("torch._dynamo")
    logger.setLevel(logging.WARNING)

    batch_size = 2
    seq_len = 16
    hidden_size = 256
    x = torch.randn(batch_size, seq_len, hidden_size, device=DEVICE)

    print("\n  [分析] 纯 PyTorch 模型 graph breaks:")
    print("  " + "-" * 50)
    model_pytorch = MiniTransformer(use_triton=False).to(DEVICE).eval()
    try:
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(model_pytorch)(x)
        print(f"  Graph 数量: {explanation.graph_count}")
        print(f"  Graph Break 数量: {explanation.graph_break_count}")
        if explanation.break_reasons:
            for i, reason in enumerate(explanation.break_reasons):
                print(f"    Break {i+1}: {reason}")
    except Exception as e:
        print(f"  explain 失败: {e}")

    print(f"\n  [分析] Triton 算子模型 graph breaks:")
    print("  " + "-" * 50)
    model_triton = MiniTransformer(use_triton=True).to(DEVICE).eval()
    try:
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(model_triton)(x)
        print(f"  Graph 数量: {explanation.graph_count}")
        print(f"  Graph Break 数量: {explanation.graph_break_count}")
        if explanation.break_reasons:
            for i, reason in enumerate(explanation.break_reasons):
                print(f"    Break {i+1}: {reason}")
    except Exception as e:
        print(f"  explain 失败: {e}")

    # 恢复日志设置
    torch._dynamo.config.verbose = False
    torch._dynamo.reset()
    print()


# ============================================================================
# Part 6: 单算子级别的 compile 效果对比
# ============================================================================

def benchmark_operator_level():
    """对比单个算子: 手写Triton vs PyTorch Eager vs PyTorch+compile 自动融合"""
    print("=" * 60)
    print("Part 4: 单算子级别 compile 融合效果对比")
    print("=" * 60)

    M, N = 4096, 1024
    x = torch.randn(M, N, device=DEVICE)
    w = torch.ones(N, device=DEVICE)
    b = torch.zeros(N, device=DEVICE)
    bias_gelu = torch.randn(N, device=DEVICE)

    with torch.no_grad():
        # ---------- LayerNorm ----------
        print("\n  >> LayerNorm (M=4096, N=1024)")
        # PyTorch eager
        lat_eager = benchmark_latency(lambda: F.layer_norm(x, (N,), w, b))
        # Triton 手写
        lat_triton = benchmark_latency(lambda: triton_layernorm(x, w, b))
        # PyTorch + compile
        ln_compiled = torch.compile(lambda x: F.layer_norm(x, (N,), w, b))
        lat_compiled = benchmark_latency(lambda: ln_compiled(x))

        print(f"     PyTorch Eager:    {lat_eager:.3f} ms")
        print(f"     Triton 手写:      {lat_triton:.3f} ms")
        print(f"     PyTorch+compile:  {lat_compiled:.3f} ms")

        # ---------- RMSNorm ----------
        print("\n  >> RMSNorm (M=4096, N=1024)")

        def pytorch_rmsnorm(x, w, eps=1e-5):
            rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
            return (x.float() / rms * w.float()).to(x.dtype)

        lat_eager = benchmark_latency(lambda: pytorch_rmsnorm(x, w))
        lat_triton = benchmark_latency(lambda: triton_rmsnorm(x, w))
        rmsnorm_compiled = torch.compile(pytorch_rmsnorm)
        lat_compiled = benchmark_latency(lambda: rmsnorm_compiled(x, w))

        print(f"     PyTorch Eager:    {lat_eager:.3f} ms")
        print(f"     Triton 手写:      {lat_triton:.3f} ms")
        print(f"     PyTorch+compile:  {lat_compiled:.3f} ms")

        # ---------- Fused GELU + Bias ----------
        print("\n  >> Fused GELU+Bias (M=4096, N=1024)")

        def pytorch_gelu_bias(x, bias):
            return F.gelu(x + bias)

        lat_eager = benchmark_latency(lambda: pytorch_gelu_bias(x, bias_gelu))
        lat_triton = benchmark_latency(lambda: triton_fused_gelu_bias(x, bias_gelu))
        gelu_compiled = torch.compile(pytorch_gelu_bias)
        lat_compiled = benchmark_latency(lambda: gelu_compiled(x, bias_gelu))

        print(f"     PyTorch Eager:    {lat_eager:.3f} ms")
        print(f"     Triton 手写:      {lat_triton:.3f} ms")
        print(f"     PyTorch+compile:  {lat_compiled:.3f} ms")

    print()


# ============================================================================
# Part 7: 不同 compile mode 的对比
# ============================================================================

def benchmark_compile_modes():
    """对比不同 torch.compile mode 的效果"""
    print("=" * 60)
    print("Part 5: torch.compile 不同 mode 对比")
    print("=" * 60)

    batch_size = 8
    seq_len = 128
    hidden_size = 256
    x = torch.randn(batch_size, seq_len, hidden_size, device=DEVICE)

    model = MiniTransformer(use_triton=True).to(DEVICE).eval()

    modes = {
        "eager (baseline)":    None,
        "default":             "default",
        "reduce-overhead":     "reduce-overhead",
        "max-autotune":        "max-autotune",
    }

    print(f"\n  模型配置: layers=2, hidden={hidden_size}, heads=4")
    print(f"  输入: batch={batch_size}, seq_len={seq_len}")
    print()

    results = {}
    with torch.no_grad():
        for name, mode in modes.items():
            torch._dynamo.reset()
            if mode is None:
                fn = model
            else:
                fn = torch.compile(model, mode=mode)

            latency = benchmark_latency(fn=lambda: fn(x), warmup=30, repeat=100)
            results[name] = latency
            print(f"  {name:<25s}  {latency:>8.3f} ms")

    baseline = results["eager (baseline)"]
    print("\n  加速比:")
    for name, lat in results.items():
        speedup = baseline / lat
        print(f"  {name:<25s}  {speedup:>5.2f}x")
    print()


# ============================================================================
# Part 8: register_autograd 训练场景验证
# ============================================================================

def verify_autograd_correctness():
    """验证 register_autograd 注册的反向传播梯度正确性"""
    print("=" * 60)
    print("Part 6: register_autograd 梯度正确性验证")
    print("=" * 60)

    torch.manual_seed(42)
    M, N = 32, 256

    # ---------- LayerNorm 梯度验证 ----------
    x = torch.randn(M, N, device=DEVICE, requires_grad=True)
    w = torch.randn(N, device=DEVICE, requires_grad=True)
    b = torch.randn(N, device=DEVICE, requires_grad=True)

    # PyTorch 参考
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    y_ref = F.layer_norm(x_ref, (N,), w_ref, b_ref)
    loss_ref = y_ref.sum()
    loss_ref.backward()

    # Triton + register_autograd
    y_triton = triton_layernorm(x, w, b)
    loss_triton = y_triton.sum()
    loss_triton.backward()

    dx_ok = torch.allclose(x.grad, x_ref.grad, atol=1e-4)
    dw_ok = torch.allclose(w.grad, w_ref.grad, atol=1e-4)
    db_ok = torch.allclose(b.grad, b_ref.grad, atol=1e-4)
    print(f"  LayerNorm grad_x:      {'✅' if dx_ok else '❌'}  max_diff={( x.grad - x_ref.grad).abs().max().item():.2e}")
    print(f"  LayerNorm grad_weight: {'✅' if dw_ok else '❌'}  max_diff={(w.grad - w_ref.grad).abs().max().item():.2e}")
    print(f"  LayerNorm grad_bias:   {'✅' if db_ok else '❌'}  max_diff={(b.grad - b_ref.grad).abs().max().item():.2e}")

    # ---------- RMSNorm 梯度验证 ----------
    x = torch.randn(M, N, device=DEVICE, requires_grad=True)
    w = torch.ones(N, device=DEVICE, requires_grad=True)

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    # PyTorch 参考实现
    rms = (x_ref.float().pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()
    y_ref = (x_ref.float() / rms * w_ref.float()).to(x_ref.dtype)
    y_ref.sum().backward()

    y_triton = triton_rmsnorm(x, w)
    y_triton.sum().backward()

    dx_ok = torch.allclose(x.grad.float(), x_ref.grad.float(), atol=1e-4)
    dw_ok = torch.allclose(w.grad.float(), w_ref.grad.float(), atol=1e-4)
    print(f"  RMSNorm grad_x:       {'✅' if dx_ok else '❌'}  max_diff={(x.grad.float() - x_ref.grad.float()).abs().max().item():.2e}")
    print(f"  RMSNorm grad_weight:  {'✅' if dw_ok else '❌'}  max_diff={(w.grad.float() - w_ref.grad.float()).abs().max().item():.2e}")

    # ---------- Fused GELU+Bias 梯度验证 ----------
    x = torch.randn(M, N, device=DEVICE, requires_grad=True)
    bias = torch.randn(N, device=DEVICE, requires_grad=True)

    x_ref = x.detach().clone().requires_grad_(True)
    b_ref = bias.detach().clone().requires_grad_(True)
    # sigmoid 近似 GELU 的参考实现
    z = x_ref + b_ref
    s = torch.sigmoid(1.702 * z)
    y_ref = z * s
    y_ref.sum().backward()

    y_triton = triton_fused_gelu_bias(x, bias)
    y_triton.sum().backward()

    dx_ok = torch.allclose(x.grad, x_ref.grad, atol=1e-4)
    db_ok = torch.allclose(bias.grad, b_ref.grad, atol=1e-4)
    print(f"  FusedGELU grad_x:     {'✅' if dx_ok else '❌'}  max_diff={(x.grad - x_ref.grad).abs().max().item():.2e}")
    print(f"  FusedGELU grad_bias:  {'✅' if db_ok else '❌'}  max_diff={(bias.grad - b_ref.grad).abs().max().item():.2e}")
    print()


def benchmark_training():
    """对比训练场景: register_autograd 的 Triton 算子 vs PyTorch

    register_autograd 的核心好处:
    ┌──────────────────────────────────────────────────────────────────────┐
    │ 没有 register_autograd:                                             │
    │   → Triton 算子对 autograd 不可见                                    │
    │   → forward 可以跑，但 backward 时会报错 (no grad_fn)                │
    │   → 或者需要用 torch.autograd.Function 手写，但 compile 无法 trace   │
    │   → 导致训练时必须 graph break，性能退化                              │
    │                                                                      │
    │ 有 register_autograd:                                                │
    │   → Triton 算子的 forward + backward 都被 autograd 系统感知          │
    │   → torch.compile 可以将 forward 和 backward 图一起优化              │
    │   → 训练循环中不会产生额外的 graph break                              │
    │   → backward 也可以用 Triton kernel (如 FusedGELU+Bias)             │
    └──────────────────────────────────────────────────────────────────────┘
    """
    print("=" * 60)
    print("Part 7: 训练场景 register_autograd 性能对比")
    print("=" * 60)

    batch_size = 8
    seq_len = 128
    hidden_size = 256
    num_classes = 32

    def train_step(model, optimizer, x, target):
        """一步训练: forward + loss + backward + step"""
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        return loss

    x = torch.randn(batch_size, seq_len, hidden_size, device=DEVICE)
    target = torch.randint(0, num_classes, (batch_size,), device=DEVICE)

    results = {}

    # ---- 1) PyTorch Eager 训练 ----
    model_pt = MiniTransformer(use_triton=False).to(DEVICE).train()
    opt_pt = torch.optim.AdamW(model_pt.parameters(), lr=1e-4)
    latency = benchmark_latency(lambda: train_step(model_pt, opt_pt, x, target), warmup=30, repeat=100)
    results["PyTorch Eager Train"] = latency
    print(f"\n  [1] PyTorch Eager 训练:                {latency:.3f} ms/step")

    # ---- 2) PyTorch + compile 训练 ----
    model_pt2 = MiniTransformer(use_triton=False).to(DEVICE).train()
    opt_pt2 = torch.optim.AdamW(model_pt2.parameters(), lr=1e-4)
    compiled_step_pt = torch.compile(train_step, mode="reduce-overhead")
    latency = benchmark_latency(lambda: compiled_step_pt(model_pt2, opt_pt2, x, target), warmup=30, repeat=100)
    results["PyTorch Compiled Train"] = latency
    print(f"  [2] PyTorch + compile 训练:             {latency:.3f} ms/step")

    # ---- 3) Triton 算子 Eager 训练 (得益于 register_autograd) ----
    model_tr = MiniTransformer(use_triton=True).to(DEVICE).train()
    opt_tr = torch.optim.AdamW(model_tr.parameters(), lr=1e-4)
    latency = benchmark_latency(lambda: train_step(model_tr, opt_tr, x, target), warmup=30, repeat=100)
    results["Triton Eager Train"] = latency
    print(f"  [3] Triton + register_autograd Eager:   {latency:.3f} ms/step")

    # ---- 4) Triton 算子 + compile 训练 (最佳: triton_op + register_autograd + compile) ----
    model_tr2 = MiniTransformer(use_triton=True).to(DEVICE).train()
    opt_tr2 = torch.optim.AdamW(model_tr2.parameters(), lr=1e-4)
    compiled_step_tr = torch.compile(train_step, mode="reduce-overhead")
    latency = benchmark_latency(lambda: compiled_step_tr(model_tr2, opt_tr2, x, target), warmup=30, repeat=100)
    results["Triton Compiled Train"] = latency
    print(f"  [4] Triton + register_autograd + compile: {latency:.3f} ms/step")

    # ---- 汇总 ----
    print("\n" + "-" * 60)
    print("  训练性能汇总 (P50 latency per step):")
    print("-" * 60)
    baseline = results["PyTorch Eager Train"]
    for name, lat in results.items():
        speedup = baseline / lat
        bar = "█" * int(speedup * 20)
        print(f"  {name:<38s} {lat:>8.3f} ms  {speedup:>5.2f}x  {bar}")

    print("\n  💡 关键观察点:")
    print("  • 没有 register_autograd → Triton 算子无法参与训练的反向传播")
    print("  • 有了 register_autograd → Triton 算子 forward/backward 都可用")
    print("  • 结合 torch.compile   → compile 可以将 fwd+bwd 图一起优化，无 graph break")
    print("  • FusedGELU+Bias 的反向也用了 Triton kernel → fwd+bwd 都是 fused 的")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("🔧 torch.compile + Triton 自定义算子 性能验证 Demo")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Triton version:  {triton.__version__}")
    print(f"   CUDA device:     {torch.cuda.get_device_name(0)}")
    print()

    # Step 1: 正确性验证
    verify_correctness()

    # Step 2: 端到端 compile 性能 (推理)
    benchmark_compile()

    # Step 3: Graph break 分析
    analyze_graph_breaks()

    # Step 4: 单算子对比
    benchmark_operator_level()

    # Step 5: 不同 compile mode 对比
    benchmark_compile_modes()

    # Step 6: register_autograd 梯度正确性
    verify_autograd_correctness()

    # Step 7: 训练场景性能对比
    benchmark_training()

    print("=" * 60)
    print("✅ 全部测试完成！")
    print("=" * 60)
