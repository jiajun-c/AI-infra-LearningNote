# Step 2: Triton Softmax Kernel

在实现 FlashAttention 之前，我们需要先掌握 **Softmax Kernel** 的实现。这是 FlashAttention 的基础构建块。

## 1. Softmax 回顾

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

**数值稳定性问题**：直接计算 `exp(x)` 可能导致溢出。

**解决方案**：减去最大值

```python
# 数值稳定的 softmax
max_x = max(x)
softmax(x_i) = exp(x_i - max_x) / sum(exp(x_j - max_x))
```

## 2. Softmax 的两个操作

实现 Softmax 需要两个步骤：

```
Step 1: 找最大值 (reduce max)
Step 2: 计算 exp 和求和 (reduce sum)
Step 3: 归一化
```

在 Triton 中，这两个操作需要特殊处理。

## 3. Triton 实现：基础版本

```python
import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(
    output_ptr,      # 输出指针
    input_ptr,       # 输入指针
    input_row_stride,  # 行步长
    output_row_stride,
    n_cols,          # 列数
    BLOCK_SIZE: tl.constexpr,  # 块大小 (编译时常量)
):
    """
    逐行计算 softmax
    每个 program instance 处理一行
    """
    # 1. 获取当前处理的行号
    row_idx = tl.program_id(0)

    # 2. 计算当前行的起始地址
    row_start = row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 3. 构造 mask (处理不满 BLOCK_SIZE 的情况)
    mask = col_offsets < n_cols

    # 4. 加载一行数据
    input_ptrs = input_ptr + row_start + col_offsets
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # 5. 计算 max (数值稳定性)
    row_max = tl.max(row, axis=0)

    # 6. 计算 exp(x - max)
    numerator = tl.exp(row - row_max)

    # 7. 计算 sum
    denominator = tl.sum(numerator, axis=0)

    # 8. 归一化
    softmax_output = numerator / denominator

    # 9. 写回结果
    output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x: torch.Tensor):
    """Host 函数"""
    assert x.is_cuda
    assert x.dim() == 2

    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    # 选择 BLOCK_SIZE 为大于 n_cols 的最小 2 的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # 启动 kernel: 每行一个 program instance
    grid = (n_rows,)

    softmax_kernel[grid](
        output,
        x,
        x.stride(0),  # 行步长
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
```

## 4. 关键概念详解

### 4.1 `tl.max` 和 `tl.sum` - 归约操作

```python
row_max = tl.max(row, axis=0)  # 沿 axis 0 归约，得到标量
denominator = tl.sum(numerator, axis=0)  # 沿 axis 0 归约
```

这些是 **block-level reduction**，在 SRAM 中完成，不需要写回 HBM。

### 4.2 `other=-float('inf')` - 填充值

```python
row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
```

当 `mask=False` 时，用 `-inf` 填充。这样在计算 `max` 时，这些位置不会影响结果（因为 `-inf` 永远不会成为最大值）。

### 4.3 `BLOCK_SIZE` 作为编译时常量

```python
BLOCK_SIZE: tl.constexpr
```

`constexpr` 意味着这个值在编译时确定，编译器可以针对特定大小进行优化（如循环展开、向量化）。

## 5. 为什么 Softmax Kernel 是基础？

在 FlashAttention 中，我们需要：

```
1. 对每个 Q block，计算与所有 K block 的注意力分数
2. 对这些分数做 softmax
3. 但我们不想存储完整的注意力矩阵!
```

Softmax Kernel 教会我们：
- 如何在 Triton 中做归约操作 (`max`, `sum`)
- 如何处理数值稳定性
- 如何使用 mask 处理变长数据

## 6. 练习

运行本目录下的 `softmax_kernel.py`，然后尝试：

1. **修改 BLOCK_SIZE**：观察对性能的影响
2. **添加调试输出**：打印 `row_max` 和 `denominator`
3. **实现 LogSoftmax**：`log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))`

## 7. 下一步

掌握了 Softmax Kernel 后，下一步学习 **分块计算 (Tiling)** 策略。

→ [Step 3: 分块计算](../step3_tiling/README.md)