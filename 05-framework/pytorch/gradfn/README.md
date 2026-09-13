# grad fn

通过grad_fn我们可以save_for_backward，需要注意的是，save_for_backward中不仅保存了tensor，还保存了其版本信息等，然后在backward阶段对数据进行unpack，检查tensor是否被修改了

```shell
ctx.save_for_backward(x, y)
   ↓  绑定在 torch/csrc/autograd/python_function.cpp（该文件不随 wheel 发布）
对每个 tensor 构造 SavedVariable(variable, is_output=false)
   ├─ data_          : tensor 或其 tensor_data
   ├─ saved_version_ : 版本号快照      → backward 时做 in-place 检测
   └─ hooks_         : 若 with saved_tensors_hooks 生效
        ↓
   PySavedVariableHooks::call_pack_hook()  →  用户的 pack_hook(t)
        ↓
   图里存的是 pack_hook 的返回值（不是原 tensor）

ctx.saved_tensors
   ↓
SavedVariable::unpack(saved_for)
   ├─ hooks_ ? call_unpack_hook() : 直接用 data_
   └─ 比对 saved_version_   ← 报 "modified by an inplace operation" 的地方
```

例子，如下所示，使用save_for_backward会检查版本信息

```python
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
```