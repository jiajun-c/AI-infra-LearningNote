# FlashAttention 学习指南

本教程将循序渐进地带你从零开始理解和实现 FlashAttention。

## 前置知识

- Triton 基础：`program_id`、`load/store`、`block` 概念
- CUDA 内存层级：SRAM (共享内存) vs HBM (全局内存)
- Attention 机制的基本原理

## 学习路径

```
Step 1: 理解问题 - 标准 Attention 的内存瓶颈
   ↓
Step 2: 基础构建块 - Softmax Kernel
   ↓
Step 3: 核心思想 - 分块计算 (Tiling)
   ↓
Step 4: 关键技术 - Online Softmax
   ↓
Step 5: 完整实现 - FlashAttention Forward
   ↓
Step 6: 进阶 - FlashAttention Backward
```

## 核心问题：为什么需要 FlashAttention？

标准 Attention 的内存复杂度是 O(N²)，因为：

```python
# 标准 Attention 实现
Q @ K.T  # [N, N] 矩阵，需要存到 HBM
softmax(...)  # [N, N] 矩阵
@ V  # [N, N] x [N, d] = [N, d]
```

**问题**：`Q @ K.T` 产生的 N×N 注意力矩阵必须先写入 HBM，再读出来做 softmax。

**FlashAttention 解决方案**：
1. 分块计算：将 Q、K、V 分成小块，在 SRAM 中完成计算
2. Online Softmax：增量更新 softmax，避免存储完整的 N×N 矩阵
3. 内存复杂度从 O(N²) 降低到 O(N)

## 文件结构

```
flashAttention/
├── README.md           # 本文件 - 学习指南
├── step1_problem/      # 理解标准 Attention 的内存瓶颈
├── step2_softmax/      # Triton Softmax Kernel
├── step3_tiling/       # 分块计算思想
├── step4_online_softmax/ # Online Softmax 算法
├── step5_flash_attn/   # 完整 FlashAttention 实现
└── step6_backward/     # 反向传播实现
```

## 开始学习

1. **[Step 1: 问题分析](./step1_problem/README.md)** - 理解为什么需要 FlashAttention
2. **[Step 2: Softmax Kernel](./step2_softmax/README.md)** - 掌握 Triton 中的归约操作
3. **[Step 3: 分块计算](./step3_tiling/README.md)** - 理解 Tiling 策略
4. **[Step 4: Online Softmax](./step4_online_softmax/README.md)** - 掌握增量 softmax 更新
5. **[Step 5: FlashAttention](./step5_flash_attn/README.md)** - 完整实现
6. **[Step 6: 反向传播](./step6_backward/README.md)** - 高效的反向传播

## 关键公式速查

### 标准 Softmax
```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
```

### Online Softmax (增量更新)
```
# 当前块: m_new = max(m_old, m_curr)
# 归一化因子更新:
exp(m_old - m_new) * l_old + exp(m_curr - m_new) * l_curr
```

### FlashAttention 分块
```
for each Q_block:
    for each K_block, V_block:
        # 在 SRAM 中计算局部 attention
        # 使用 online softmax 增量更新
```

## 参考资料

- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691)
- [Triton 官方 FlashAttention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)