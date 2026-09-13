"""save_for_backward 的 in-place 检查：最小复现。

save_for_backward(x) 会把 x 当时的 version 记进 SavedVariable.saved_version_，
backward 时 unpack() 拿出来比对，不一致就报错。
"""
import torch


class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)      # 记下 x 此刻的版本号
        return x * x

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors      # ← 版本号在这里被比对
        return g * 2 * x

class Square_equal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.x = x      # 记下 x 此刻的版本号
        return x * x

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.x      # ← 版本号在这里被比对
        return g * 2 * x

def case(modified):
    x = torch.ones(1, requires_grad=True)
    y = Square_equal.apply(x)
    v0 = x._version
    if modified:
        with torch.no_grad():
            x.add_(1)                 # 原地修改 → 版本号 +1
    print(f"  forward 时 version={v0}，现在 version={x._version}")
    try:
        y.backward()
        print("  → backward 通过")
    except RuntimeError as e:
        print("  → RuntimeError:", str(e).split(";")[0])


print("=== 不修改 x ===")
case(False)

print("=== 修改 x（no_grad 下 add_）===")
case(True)

# 注意：用 x.data.add_(1) 改不会 bump 版本号，检查会被绕过 —— 梯度静默算错。
# 这正是 .data 危险的地方：它给你一个看起来正常的错误结果，而不是报错。
