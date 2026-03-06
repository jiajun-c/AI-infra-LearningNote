# Expert Parallel (EP)

Expert Parallel 是 MoE（Mixture of Experts）模型中的一种并行策略。

## 1. MoE 基础

在 MoE 模型中，每个 MoE 层包含多个"专家"（Expert）网络和一个门控网络（Gating Network）。

```
输入 -> 门控网络 -> 选择 Top-K 专家 -> 专家处理 -> 加权输出
```

## 2. Expert Parallel 原理

EP 将不同的专家分配到不同的设备上：

- 每个设备持有部分专家
- 输入 token 根据门控决策被路由到对应设备
- 各设备并行处理分配给自己的 token

### 2.1 与 Tensor Parallel 对比

| 特性 | Tensor Parallel | Expert Parallel |
|------|----------------|-----------------|
| 切分维度 | 矩阵维度 | 专家维度 |
| 通信开销 | 每层都需通信 | 仅在路由时通信 |
| 扩展性 | 受限于单层大小 | 可扩展到更多设备 |

## 3. 实现要点

### 3.1 Token 路由

```python
# 简化的路由逻辑
def route_tokens(tokens, gate_output, num_experts, devices):
    # 选择 Top-K 专家
    top_k_experts = torch.topk(gate_output, k=2, dim=-1)

    # 根据专家分配将 token 发送到对应设备
    # ...
```

### 3.2 负载均衡

需要确保各设备间的 token 分布相对均匀：

- 使用容量因子（Capacity Factor）
- 动态调整 batch 大小
- 辅助损失函数促进负载均衡

## 4. 混合并行策略

实际应用中，EP 通常与其他并行策略结合使用：

```
数据并行 (DP) × Expert 并行 (EP) × Tensor 并行 (TP) × 流水线并行 (PP)
```

## 5. 通信模式

### 5.1 All-to-All 通信

EP 中主要使用 All-to-All 通信进行 token 路由：

```python
import torch.distributed as dist

# 将 token 发送到对应专家所在的设备
dist.all_to_all_single(output, input, group=ep_group)
```

## 6. 挑战与优化

### 6.1 负载不均衡

专家间的负载可能不均衡，导致某些设备空闲。

**解决方案**：
- 动态容量调整
- 辅助负载均衡损失

### 6.2 通信瓶颈

All-to-All 通信可能成为瓶颈。

**解决方案**：
- 梯度累积减少通信频率
- 使用高速互联（NVLink, InfiniBand）
