"""
torch.autograd.Function 实战：手写 FFN(前馈网络) 的前向与反向

    y = GELU(x @ W1 + b1) @ W2 + b2

自定义 Function 的三条铁律：

1. forward / backward 必须是 @staticmethod，第一个参数是 ctx（合并式写法）。
   不需要也不该写 __init__：实例由 autograd 引擎创建，自己 new 会触发
   DeprecationWarning（PyTorch 2.9 实测），且在将来的版本里会直接报错。
   调用方式只有一种：FFN.apply(...)，不要直接调 forward。
2. backward 返回的梯度个数必须与 forward 的输入个数严格一致，
   非张量输入（如 shape、bool、str）用 None 占位。
3. 反向实现务必用 torch.autograd.gradcheck 验证，不能靠「看起来对」。

PyTorch >= 2.0 推荐「分离式」写法（forward + setup_context），见 FFNSeparate：
只有分离式写法才能被 torch.func（vmap / grad / jacrev）等 functorch 变换识别。

运行：
    python demo.py            # 正确性验证
    python demo.py --bench    # 额外跑一次性能对比（需要 CUDA 或耐心）
"""

from __future__ import annotations

import argparse
import math
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 0. 为什么原来那段代码跑不起来
# ---------------------------------------------------------------------------


def demo_why_wrong() -> None:
    """逐个复现 `class FFN(torch.autograd.Function): def __init__(...)` 的三个坑。"""

    class BadFFN(torch.autograd.Function):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    print("=" * 72)
    print("0. 原写法的三个坑")
    print("=" * 72)

    # 坑 1：实例化 —— Function 的实例由引擎创建，不该自己 new
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bad = BadFFN()
    for w in caught:
        print(f"  [实例化] {w.category.__name__}: {str(w.message)[:88]}...")

    # 坑 2：像调用函数一样调用实例 —— 旧式（非 static forward）写法已被废弃
    try:
        bad(torch.randn(2, 4))
    except RuntimeError as e:
        print(f"  [调用实例] RuntimeError: {str(e).splitlines()[0]}")

    # 坑 3：真正致命的问题 —— 没有实现 staticmethod forward，apply 必然失败
    try:
        BadFFN.apply(torch.randn(2, 4))
    except NotImplementedError as e:
        print(f"  [apply] NotImplementedError: {e}")

    print("\n  结论：__init__ 里能做的事，ctx / setup_context 都能做；")
    print("        而 forward、backward 是必须实现的，跟 __init__ 无关。\n")


# ---------------------------------------------------------------------------
# 1. 激活函数：GELU 及其导数（必须手写，forward 内部不能依赖 autograd）
# ---------------------------------------------------------------------------


def gelu(x: torch.Tensor) -> torch.Tensor:
    """精确版 GELU: 0.5 * x * (1 + erf(x / sqrt(2)))，与 F.gelu 默认实现一致。"""
    return 0.5 * x * (1.0 + torch.erf(x * math.sqrt(0.5)))


def gelu_grad(x: torch.Tensor) -> torch.Tensor:
    """GELU 的导数: Φ(x) + x * φ(x)，其中 Φ 是标准正态 CDF、φ 是 PDF。"""
    cdf = 0.5 * (1.0 + torch.erf(x * math.sqrt(0.5)))
    pdf = torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    return cdf + x * pdf


def _flat(t: torch.Tensor) -> torch.Tensor:
    """[..., F] -> [N, F]。

    参数的梯度要对所有前导维度求和，而 x 可能是 1-D（单样本 [I]）/ 2-D（[B, I]）/
    3-D（[B, T, I]）。统一拍平成 2D 再算，反向公式就不必分情况讨论。

    这同时也是个真会踩的坑：若直接写 grad_y.sum(dim=0)，1-D 输入下 grad_y 形状是
    [O]，求和会退化成标量，b2 的梯度形状就错了；`t.mT` 对 1-D 张量更是直接报错
    （tensor.mT is only supported on matrices）。
    """
    return t.reshape(-1, t.shape[-1])


# ---------------------------------------------------------------------------
# 2. 合并式写法：forward(ctx, ...) + backward(ctx, ...)
# ---------------------------------------------------------------------------


class FFN(torch.autograd.Function):
    """y = GELU(x @ W1 + b1) @ W2 + b2

    前向内部处于 no_grad 状态，不会被 autograd 记录，
    因此所有中间结果如果反向要用，必须显式 save_for_backward。
    """

    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2):  # type: ignore[override]
        pre = x @ w1 + b1  # [..., H]   线性层 1
        h = gelu(pre)  # [..., H]   激活
        y = h @ w2 + b2  # [..., O]   线性层 2
        # 只保存反向真正需要的东西。b1/b2 的梯度是求和，无需保存。
        ctx.save_for_backward(x, w1, pre, h, w2)
        return y

    @staticmethod
    def backward(ctx, grad_y):  # type: ignore[override]
        x, w1, pre, h, w2 = ctx.saved_tensors

        # ---- 线性层 2 的反向 ----
        grad_w2 = _flat(h).mT @ _flat(grad_y)  # [H, O]
        grad_b2 = _flat(grad_y).sum(dim=0)  # [O]

        # ---- 激活的反向 ----
        grad_h = grad_y @ w2.mT  # [..., H]
        grad_pre = grad_h * gelu_grad(pre)  # [..., H]  逐元素乘

        # ---- 线性层 1 的反向 ----
        flat_grad_pre = _flat(grad_pre)
        grad_w1 = _flat(x).mT @ flat_grad_pre  # [I, H]
        grad_b1 = flat_grad_pre.sum(dim=0)  # [H]
        grad_x = grad_pre @ w1.mT  # [..., I]

        # 顺序、个数必须与 forward 的输入 (x, w1, b1, w2, b2) 一一对应
        return grad_x, grad_w1, grad_b1, grad_w2, grad_b2


# ---------------------------------------------------------------------------
# 3. 分离式写法（PyTorch >= 2.0 推荐）：forward + setup_context
# ---------------------------------------------------------------------------


class FFNSeparate(torch.autograd.Function):
    """与 FFN 等价，但采用 forward / setup_context 分离的写法。

    分离式的限制：forward 拿不到 ctx，setup_context 又只能看到 inputs 和 output，
    所以中间结果 pre / h 有两种处理方式：

      方案 A（本例）：什么都不存，backward 里用输入重算一遍 —— 省前向显存、多算力，
                     正是 gradient checkpointing 的思想（见 05-framework/pytorch/recompute/）。
                     注意：能不能省下整个训练步的显存峰值，取决于「中间结果」和
                     「反向自身临时量」谁更大，见本文 bench 的实测结论。
      方案 B：让 forward 把中间结果一并返回，setup_context 里存下来 ——
              `return y, pre, h` / `ctx.save_for_backward(x, w1, pre, h, w2)` /
              `def backward(ctx, grad_y, grad_pre, grad_h): ...`，
              调用方写 `y, _, _ = FFNSeparate.apply(...)`。省算力、费显存。

    收益：只有分离式写法才能被 torch.func(vmap / grad / jacrev) 识别。
    """

    # 允许 torch.vmap 自动生成规则（前提：forward/backward 全是可 vmap 的 torch 算子）
    generate_vmap_rule = True

    @staticmethod
    def forward(x, w1, b1, w2, b2):  # type: ignore[override]
        return gelu(x @ w1 + b1) @ w2 + b2

    @staticmethod
    def setup_context(ctx, inputs, output):  # type: ignore[override]
        x, w1, b1, w2, _b2 = inputs
        ctx.save_for_backward(x, w1, b1, w2)  # 不存 pre / h

    @staticmethod
    def backward(ctx, grad_y):  # type: ignore[override]
        x, w1, b1, w2 = ctx.saved_tensors

        pre = x @ w1 + b1  # 重算：显存换算力。b1 必须存下来，否则重算不出来
        h = gelu(pre)

        grad_w2 = _flat(h).mT @ _flat(grad_y)
        grad_b2 = _flat(grad_y).sum(dim=0)

        grad_h = grad_y @ w2.mT
        grad_pre = grad_h * gelu_grad(pre)

        flat_grad_pre = _flat(grad_pre)
        grad_w1 = _flat(x).mT @ flat_grad_pre
        grad_b1 = flat_grad_pre.sum(dim=0)
        grad_x = grad_pre @ w1.mT

        return grad_x, grad_w1, grad_b1, grad_w2, grad_b2


# ---------------------------------------------------------------------------
# 4. 封装成 nn.Module：可训练、可进 optimizer、可存 state_dict
# ---------------------------------------------------------------------------


class _FFNParams(nn.Module):
    """参数容器：自定义版与参考版共用同一套参数和初始化。"""

    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(in_features, hidden))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.empty(hidden, out_features))
        self.b2 = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 与 nn.Linear.reset_parameters 保持一致，便于和参考实现逐位对比
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        nn.init.uniform_(self.b1, -1 / math.sqrt(self.w1.shape[0]), 1 / math.sqrt(self.w1.shape[0]))
        nn.init.uniform_(self.b2, -1 / math.sqrt(self.w2.shape[0]), 1 / math.sqrt(self.w2.shape[0]))


class FFNModule(_FFNParams):
    """自定义 Function 版 FFN。"""

    def __init__(self, in_features: int, hidden: int, out_features: int, fn=FFN):
        super().__init__(in_features, hidden, out_features)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn.apply(x, self.w1, self.b1, self.w2, self.b2)


class RefFFN(_FFNParams):
    """参考实现：完全交给 autograd 组合算子，没有自定义反向。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x @ self.w1 + self.b1) @ self.w2 + self.b2


# ---------------------------------------------------------------------------
# 5. 正确性验证
# ---------------------------------------------------------------------------


def check_gradcheck() -> None:
    """数值梯度 vs 解析梯度：反向写错了这里一定会炸。必须用 float64。"""
    print("=" * 72)
    print("1. gradcheck：用数值微分校验手写反向")
    print("=" * 72)

    torch.manual_seed(0)
    B, I, H, O = 4, 6, 8, 3
    kw = dict(dtype=torch.float64, requires_grad=True)
    args = (
        torch.randn(B, I, **kw),
        torch.randn(I, H, **kw),
        torch.randn(H, **kw),
        torch.randn(H, O, **kw),
        torch.randn(O, **kw),
    )

    for name, fn in (("FFN(合并式)", FFN), ("FFNSeparate(分离式)", FFNSeparate)):
        ok = torch.autograd.gradcheck(fn.apply, args, eps=1e-6, atol=1e-5, rtol=1e-3)
        print(f"  {name:<22} gradcheck = {ok}")

    # gradcheck 只校验一阶导；如果要支持二阶导，需要 backward 内部也用可微算子
    # 并保存带 grad_fn 的张量，否则应显式标记 @once_differentiable 让二次反向直接报错
    # （而不是悄悄给出错误结果）。
    print()


def check_against_autograd() -> None:
    """自定义反向 vs autograd 自动生成的反向，逐参数比对。"""
    print("=" * 72)
    print("2. 与 autograd 参考实现比对（输出 + 全部参数梯度）")
    print("=" * 72)

    torch.manual_seed(0)
    B, I, H, O, T = 8, 16, 32, 4, 5

    for fn_name, fn in (("FFN(合并式)", FFN), ("FFNSeparate(分离式)", FFNSeparate)):
        custom = FFNModule(I, H, O, fn=fn).double()
        ref = RefFFN(I, H, O).double()
        ref.load_state_dict(custom.state_dict())  # 保证权重完全一致

        # 3D 输入，顺带验证 backward 里 reshape 处理前导维度的正确性
        x = torch.randn(B, T, I, dtype=torch.float64)
        y_custom = custom(x)
        y_ref = ref(x)

        g = torch.randn_like(y_custom)
        y_custom.backward(g)
        y_ref.backward(g)

        print(f"  {fn_name}")
        print(f"    前向最大误差 : {(y_custom - y_ref).abs().max().item():.3e}")
        for name, p in custom.named_parameters():
            diff = (p.grad - ref.get_parameter(name).grad).abs().max().item()
            print(f"    grad {name:<4}  最大误差 : {diff:.3e}")
    print()


def check_functorch() -> None:
    """分离式写法才吃得到 torch.func 的变换：这是 PyTorch 2.x 的推荐理由。"""
    print("=" * 72)
    print("3. torch.func 兼容性（vmap / grad）")
    print("=" * 72)

    from torch.func import grad, vmap

    torch.manual_seed(0)
    I, H, O = 6, 8, 3
    x = torch.randn(4, I, dtype=torch.float64)
    w1 = torch.randn(I, H, dtype=torch.float64)
    b1 = torch.randn(H, dtype=torch.float64)
    w2 = torch.randn(H, O, dtype=torch.float64)
    b2 = torch.randn(O, dtype=torch.float64)

    def loss_combined(xi):
        return FFN.apply(xi, w1, b1, w2, b2).sum()

    try:
        grad(loss_combined)(x[0])
        print("  合并式 + torch.func.grad : 居然通过了（PyTorch 版本差异）")
    except RuntimeError as e:
        print(f"  合并式 + torch.func.grad : RuntimeError -> {str(e).splitlines()[0][:70]}...")

    def loss_separate(xi):
        return FFNSeparate.apply(xi, w1, b1, w2, b2).sum()

    g_sample = grad(loss_separate)(x[0])
    print(f"  分离式 + torch.func.grad : ok, 单样本梯度 shape = {tuple(g_sample.shape)}")

    # 逐样本梯度（等价于 functorch 的 per-sample-grad），合并式写法做不到
    per_sample = vmap(grad(loss_separate))(x)
    print(f"  分离式 + vmap(grad)      : ok, per-sample 梯度 shape = {tuple(per_sample.shape)}")


def check_train_step() -> None:
    """放进真实训练循环：只有梯度对，loss 才会降。"""
    print()
    print("=" * 72)
    print("4. 端到端训练一步")
    print("=" * 72)

    torch.manual_seed(0)
    model = FFNModule(16, 32, 1)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    x = torch.randn(64, 16)
    target = torch.randn(64, 1)

    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = F.mse_loss(model(x), target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"  loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"  参数梯度非零: {model.w1.grad.abs().sum().item() > 0}")
    print()


def bench(device: str = "cuda", iters: int = 200) -> None:
    """算力 vs 显存的权衡：三种写法的耗时与峰值显存。"""
    print("=" * 72)
    print(f"5. 性能/显存对比（{device}, {iters} iters）")
    print("=" * 72)

    if device == "cuda" and not torch.cuda.is_available():
        print("  无可用 CUDA，跳过")
        return

    torch.manual_seed(0)
    B, I, H, O = 4096, 1024, 4096, 1024
    x = torch.randn(B, I, device=device)

    cases = [
        ("RefFFN  (autograd+F.gelu)", RefFFN(I, H, O)),
        ("FFN     (保存中间结果)   ", FFNModule(I, H, O, fn=FFN)),
        ("FFNSep  (反向重计算)     ", FFNModule(I, H, O, fn=FFNSeparate)),
    ]
    cases = [(name, m.to(device)) for name, m in cases]
    for _, m in cases[1:]:
        m.load_state_dict(cases[0][1].state_dict())

    def step(model):
        model.zero_grad(set_to_none=True)
        y = model(x)
        y.sum().backward()

    def timeit(model):
        for _ in range(20):  # warmup
            step(model)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            step(model)
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    def peak_mem(model):
        """整个训练步的显存峰值。

        注意：只测「整步」峰值，不单独测前向峰值 —— 后者受 CUDA caching
        allocator 复用行为影响，同一份代码两次运行能差出几百 MiB，不可复现。
        """
        if device != "cuda":
            return float("nan")
        for _ in range(5):
            step(model)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        step(model)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2**20

    base_t = None
    for name, model in cases:
        t = timeit(model)
        base_t = base_t or t
        full = peak_mem(model)
        print(f"  {name} : {t:8.3f} ms/iter ({t / base_t:4.2f}x)   整步峰值显存 {full:6.1f} MiB")

    print("  结论 1：本例的自定义反向只是「逐算子复刻」，所以不会更快 ——")
    print("          手写的 GELU 被拆成 erf/mul/exp 一串逐元素 kernel，而 F.gelu 是单个")
    print("          融合 kernel；eager 模式下我们也无法把 matmul 和 elementwise 融合。")
    print("          自定义 Function 的价值在于写出 autograd 组合不出来的东西（自定义")
    print("          CUDA kernel、数值稳定公式、非标准算子），而不是把同样的算子抄一遍。")
    print("  结论 2：这个规模下自定义 Function 也没省显存 —— FFN 比 RefFFN 高，是因为")
    print("          backward 里 gelu_grad 展开出的临时张量、加上保存的 pre/h 同时存活；")
    print("          FFNSep 省掉了 forward 的保存，但 backward 重算的 pre/h 又补了回来，")
    print("          净效果约等于 0。重计算要真能省显存，前提是「中间结果 ≫ 反向临时量」")
    print("          且网络足够深（见 05-framework/pytorch/recompute/ 里的 checkpoint 对比），")
    print("          两层 MLP 不满足这个前提。别把「手写 Function / 重计算更省显存」")
    print("          当成无条件结论，实测才算数。")


def main() -> None:
    parser = argparse.ArgumentParser(description="torch.autograd.Function 手写 FFN")
    parser.add_argument("--bench", action="store_true", help="额外跑性能对比")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"PyTorch {torch.__version__} | device = {args.device}\n")
    demo_why_wrong()
    check_gradcheck()
    check_against_autograd()
    check_functorch()
    check_train_step()
    if args.bench:
        bench(args.device)


if __name__ == "__main__":
    main()
