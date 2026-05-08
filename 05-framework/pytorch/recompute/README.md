# torch重计算

https://pytorch.org/blog/activation-checkpointing-techniques/

torch的重计算机制是为了节省训练时的显存，在默认的情况下前向的过程中会存储下每一层的激活值。会导致显存值爆增

```python3
python3 example.py
```

根据这个输出，在重计算的时候，前向的激活值不被存储

```shell
初始化输入张量...
=== 标准模式 - 前向与反向传播 ===
当前常驻显存: 1137.63 MB
峰值占用显存: 3041.32 MB
计算过程额外消耗峰值: 2497.01 MB

=== 重计算模式 - 前向与反向传播 ===
当前常驻显存: 1138.63 MB
峰值占用显存: 1346.63 MB
计算过程额外消耗峰值: 737.31 MB
```

## 选择性重计算（手动）

`example_selective.py` 演示三种手动策略对比：无重计算 / 全量重计算 / 每隔 k 层重计算一次。

```python3
python3 example_selective.py
```

## 自动激活检查点（torch.compile + activation_memory_budget）

`example_auto_budget.py` 演示 `torch.compile` 的自动激活检查点机制。

通过 `torch._functorch.config.activation_memory_budget` 控制显存预算（0.0～1.0），编译器在 AOT Autograd 分区阶段用 0-1 背包算法自动决定哪些算子丢弃激活、反向时重算：
- **1.0**：完全保留激活，不重计算（默认，最快）
- **中间值**：在显存预算约束下，优先重算"便宜"的算子（gelu、elementwise 等），保留 matmul 等昂贵算子的激活；越小显存越省，重计算开销越大
- **0.0**：特殊分支（源码 `partitioners.py`），只保存图输入节点（含模型参数权重），中间激活全部重计算。由于参数权重本身体积大，峰值显存**不降反升**，与手动 `checkpoint`（只丢弃激活）行为不同，**实际不要使用 0.0**

```python3
python3 example_auto_budget.py
```

实测输出（batch=4, seq=1024, hidden=1024, layers=16）：

```shell
  策略                                    峰值显存        耗时        显存节省
  ------------------------------------------------------------------
  budget=1.0  完全保留激活                 3520.6M    0.070s      +0.0M (0.0%)
  budget=0.9                         2560.6M    0.070s    +960.0M (27.3%)
  budget=0.8                         2560.6M    0.070s    +960.0M (27.3%)
  budget=0.7                         2560.6M    0.070s    +960.0M (27.3%)
  budget=0.6                         2560.6M    0.070s    +960.0M (27.3%)
  budget=0.5                         2544.8M    0.071s    +975.9M (27.7%)
  budget=0.4                         2353.3M    0.074s   +1167.3M (33.2%)
  budget=0.3                         2209.4M    0.077s   +1311.2M (37.2%)
  budget=0.2                         2017.5M    0.079s   +1503.2M (42.7%)
  budget=0.1                         1873.5M    0.081s   +1647.1M (46.8%)
  budget=0.0  全部重计算                  3520.6M    0.204s      +0.0M (0.0%)
```

> `budget=0.0` 峰值与 `1.0` 相同，原因：源码中该分支返回 `node_info.inputs`（图输入节点，包含模型参数权重），中间激活全部重计算，但参数权重本身体积大，抵消了激活节省的收益。**实际使用建议区间为 (0.1, 0.9)**，从 0.5 开始调参。

与手动 `torch.utils.checkpoint` 的对比：

| | 手动 checkpoint | activation_memory_budget |
|---|---|---|
| 粒度 | block 级 | 算子级 |
| 需要修改模型代码 | 是 | 否 |
| 决策依据 | 人工判断 | 编译器基于 flop/memory 自动优化 |
| 适用条件 | 任何 PyTorch | 需要 `torch.compile` |

## `torch._functorch.config` 其他重要特性

该模块是 AOT Autograd 的全局配置中心，除 `activation_memory_budget` 外还有以下值得关注的参数：

### 重计算细粒度控制（ban_recompute_* 系列）

这组 bool flag 控制分区器在做激活检查点时哪些算子"禁止被选为重计算"：

| 配置项 | 默认 | 含义 |
|---|---|---|
| `ban_recompute_used_far_apart` | `True` | 禁止重计算"距离反向用点很远"的节点，避免长链重计算 |
| `ban_recompute_long_fusible_chains` | `True` | 禁止重计算过长的可融合算子链，防止反向重计算链过深 |
| `ban_recompute_materialized_backward` | `True` | 禁止重计算反向中必须物化的节点（被非融合算子使用的节点） |
| `ban_recompute_not_in_allowlist` | `True` | 只允许重计算白名单内的算子（elementwise、激活函数等），其余禁止 |
| `ban_recompute_reductions` | `True` | 禁止重计算 reduction（结果小但重算代价高） |
| `recompute_views` | `False` | 是否重计算 view 算子（view 本身免费，建议保持 True 等价的默认行为） |

全部关闭 ban_recompute 限制可用 `aggressive_recomputation = True`，会大幅增加重计算覆盖面，牺牲性能换显存。

### activation_memory_budget 求解器

```python
# 运行时代价估算方式
activation_memory_budget_runtime_estimator = "flops"   # 默认：用 flop count 估算
# 可选 "profile"（实际 benchmark 每个算子）或 "testing"（全部返回 1）

# 背包求解算法
activation_memory_budget_solver = "dp"   # 默认：量化 DP（推荐）
# 可选 "greedy"（贪心，速度快但质量差）或 "ilp"（整数线性规划，需 scipy）
```

### 调试与可视化

```python
# 打印分区器调试信息
debug_partitioner = True   # 或设环境变量 AOT_PARTITIONER_DEBUG=1

# 生成 memory budget vs recompute runtime 的 Pareto 前沿 SVG 图
# 帮助选择合适的 budget 值
visualize_memory_budget_pareto = True   # 或 PARTITIONER_MEMORY_BUDGET_PARETO=1
memory_budget_pareto_dir = "/tmp/pareto"   # SVG 输出目录
```

### 其他实用配置

```python
# 激活 aggressive 重计算模式：关闭大部分 ban_recompute 限制
# 效果类似把 budget 调得很低，但不受背包约束，可能节省更多显存
aggressive_recomputation = False   # 默认关闭

# 参数权重视为"免费保存"（不计入显存预算）
# 关闭后分区器会尝试重计算参数相关节点，通常不建议改动
treat_parameters_as_free_to_save = True

# 分区前对图做公共子表达式消除，减少冗余算子
cse = True

# 多机训练时禁止优化 collective 算子（allreduce 等），避免不同 rank 决策不一致导致 NCCL hang
unsafe_allow_optimization_of_collectives = False   # 必须保持 False
```

### 使用方式

```python
from torch._functorch import config as functorch_config

# 必须在 torch.compile 之前设置
functorch_config.activation_memory_budget = 0.3
functorch_config.activation_memory_budget_runtime_estimator = "flops"
functorch_config.activation_memory_budget_solver = "dp"
functorch_config.aggressive_recomputation = False

model = torch.compile(model)
```