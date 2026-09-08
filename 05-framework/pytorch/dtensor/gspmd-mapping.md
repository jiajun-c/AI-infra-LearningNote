# GSPMD 在 PyTorch DTensor 中的对应实现

> 调研范围：[torch.distributed.tensor](https://pytorch.org/docs/stable/distributed.tensor.html) 在 PyTorch 2.9.1 中的源码
> 调研目标：把 GSPMD 论文 (arXiv 2105.04663v2) 的设计概念映射到 PyTorch DTensor 的具体类/方法上

本文按 GSPMD 论文的结构逐条对应到 DTensor 源码（所有行号都基于 `torch.distributed.tensor` 包，相对路径 `site-packages/torch/distributed/tensor/`），帮助在阅读 GSPMD 论文的同时理解 PyTorch 的工程取舍。

---

## 0. 总览：一表看懂 GSPMD ↔ DTensor

| GSPMD 概念 | PyTorch DTensor 对应 | 源码位置 |
|---|---|---|
| `mesh_split(t, mesh, dims_mapping)` | `Shard(dim)` + `DeviceMesh` | `placement_types.py:51`, `device_mesh.py:395` |
| Replicated | `Replicate()` | `placement_types.py:602` |
| Partially tiled（先分组后切分） | `Partial(reduce_op)` | `placement_types.py:654` |
| `device_mesh`（N 维逻辑网格） | `DeviceMesh` 类 | `device_mesh.py:395` |
| `dims_mapping[i] = j / -1` | `DTensorSpec.dim_map` | `_dtensor_spec.py:146` |
| `DTensorSpec`（张量的全局描述） | `DTensorSpec` dataclass | `_dtensor_spec.py:25` |
| Sharding propagation（按算子规则补全） | `ShardingPropagator` | `_sharding_prop.py:52` |
| Per-op partition rule（如 Einsum） | `register_op_strategy` 注册的 `OpStrategy` | `_ops/_matrix_ops.py`, `_ops/_common_rules.py` |
| AllReduce / AllGather / AllToAll 插入 | `redistribute_local_tensor` | `_redistribute.py:157` |
| Resharding（注解变化触发通信） | `Redistribute` autograd Function | `_redistribute.py:281` |
| Halo exchange（Conv 邻域数据） | `_conv_ops.py` 内的卷积策略 | `_ops/_conv_ops.py` |
| 数据并行 / 模型并行 / 优化器状态分片 | `parallel/ddp.py` + `parallel/style.py` + `parallel/fsdp.py` | `parallel/` |
| 嵌套并行（partial tiling） | `Partial` + `Shard` 同列共存 | `placement_types.py:170-189` |
| `_StridedShard`（FSDP+TP 右到左分片） | `_StridedShard(Shard)` 子类 | `placement_types.py:367` |

---

## 1. DeviceMesh：N 维逻辑进程网格

**GSPMD（论文 §3.1）**：用 `device_mesh` 把 N 个 device 组织成多维逻辑网格，例如 `(data, model) = (2, 4)`。每个 mesh 维独立持有一个子进程组。

**DTensor 实现**：`torch.distributed.device_mesh.DeviceMesh`

```python
# 初始化
mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))
```

源码关键点（[`device_mesh.py:444-489`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/device_mesh.py)）：

```python
def __init__(self, device_type, mesh, *, mesh_dim_names=None, ...):
    # mesh 是一个 int tensor，描述设备在 mesh 各维上的全局 rank 分布
    self.mesh = torch.tensor(mesh, dtype=torch.int)
    self.mesh_dim_names = tuple(mesh_dim_names) if mesh_dim_names else None
    # 为每个 mesh 维自动创建一个子 process group
    self._init_process_groups(backend_override)
    # 当前 rank 在 mesh 中的坐标
    self._coordinate_on_dim = (self.mesh == get_rank()).nonzero()[0].tolist()
```

`_init_process_groups`（[`device_mesh.py:541-666`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/device_mesh.py)）的精髓：

- 对每个 mesh 维 `d`，把 mesh reshape 成 `(mesh.size(d), -1)`，得到 d 上的子进程组。
- 如果 NCCL 支持 `bound_device_id`，优先用 `split_group`（一次调用分裂出所有子组）；否则回退到逐子组 `new_group`。
- 每个 mesh 维独立 group，对应 GSPMD 论文里"每个 mesh dim 一个 sub-Pattern"的描述。

**子网格（sub-mesh）**：GSPMD 允许用户只使用 mesh 的一个子集做并行。DTensor 通过 `__getitem__` 实现（[`device_mesh.py:718`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/device_mesh.py)）：

```python
tp_mesh = mesh_2d["tp"]        # 1D sub-mesh
dp_tp_mesh = mesh_3d["dp", "tp"] # 2D sub-mesh
```

**与 GSPMD 的差异**：DTensor 的 mesh 不强制要求是 `data × model` 矩形——可以用任意 `torch.Tensor` 描述异构拓扑（如 `[0,1,2,3], [4,5,6,7]` 表示两台 host 各 4 卡）。这比 GSPMD 的规则 mesh 更灵活，但网格内通信顺序仍按"沿 dim 切一刀"的逻辑进行。

---

## 2. Placement：GSPMD 三种基础分片的 PyTorch 化

**GSPMD（论文 §3.1, Figure 1）** 定义了三种基础分片：Replicated / Tiled / Partially tiled。

**DTensor 实现**：[`torch.distributed.tensor.placement_types`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/placement_types.py)：

```python
class Placement: ...                                  # 抽象基类
class Shard(Placement): dim: int                      # = GSPMD Tiled
class Replicate(Placement): ...                       # = GSPMD Replicated
class Partial(Placement): reduce_op: str = "sum"      # = GSPMD Partially tiled
class _StridedShard(Shard): split_factor: int         # = GSPMD 部分 mesh 顺序的扩展
```

### 2.1 `Shard(dim)` ↔ GSPMD Tiled

[`placement_types.py:50-360`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/placement_types.py) 的核心：

- `_split_tensor` 用 `torch.chunk` 把张量在 `dim` 上切 `num_chunks` 份，并处理**非均匀分片**的 padding（`with_padding=True`）。
- `_local_shard_size_and_offset` 计算当前 rank 的本地 shard 大小与偏移（处理 `tensor_size % num_chunks != 0` 的边界情况）——直接对应 GSPMD §4.1 的 `pad_tensor + DynamicSlice` 静态 shape 约束。
- `_shard_tensor` 用 `mesh_scatter`（沿 mesh 维 0 号 rank 为 source）做 `scatter`；`_reduce_shard_tensor` 用 `reduce_scatter_tensor`（对应 GSPMD §4.2 的 ReduceScatter）。
- `_to_replicate_tensor` 用 `all_gather_tensor` 沿 shard 维收集（对应 GSPMD §4.2 的 AllGather）。
- `_to_new_shard_dim` 在跨 mesh 维 shard 转移时插入 `alltoall`（对应 GSPMD §4.2 的 AllToAll）。
- `_replicate_to_shard` 走纯本地 `chunk + clone`（不需要通信）。

### 2.2 `Replicate()` ↔ GSPMD Replicated

[`placement_types.py:602-651`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/placement_types.py)：

```python
def _replicate_tensor(self, tensor, mesh, mesh_dim, src_data_rank=0):
    # 用 mesh_broadcast 把 src_data_rank 的张量广播到该 mesh 维上
    mesh_broadcast(tensor, mesh, mesh_dim=mesh_dim, group_src=src_data_rank)
    return tensor
```

### 2.3 `Partial(reduce_op)` ↔ GSPMD Partially tiled

[`placement_types.py:654-720`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/placement_types.py)——**这是 DTensor 相对 GSPMD 论文的一个工程增强**：

```python
class Partial(Placement):
    reduce_op: str = "sum"

    def _reduce_value(self, tensor, mesh, mesh_dim):
        # Partial -> Replicate：执行 AllReduce
        return funcol.all_reduce(tensor, reduceOp=self.reduce_op,
                                 group=(mesh, mesh_dim))

    def _reduce_shard_value(self, tensor, mesh, mesh_dim, shard_spec):
        # Partial -> Shard(dim)：执行 ReduceScatter
        return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)

    def _partition_value(self, tensor, mesh, mesh_dim):
        # Replicate -> Partial(sum)：除以 mesh 维大小
        assert self.reduce_op == "sum", "only support replicate to PartialSUM"
        return tensor / mesh.size(mesh_dim)
```

GSPMD 论文里 "partial sum" 只在输出端隐式产生。DTensor 把 `Partial` 显式提升为第一类 placement：
- `f(a) = a; f(a+b) = f(a) + f(b)` 的**线性算子**（如 add、matmul）可以让 `Partial` 沿算子**透传**，把通信推迟到下游 reduce 处合并——典型的例子就是 mm：`S(0) @ S(1)` 的输出是 `Partial`，而不是 `Replicate`。
- `Partial` 还支持 `is_partial(reduce_op="sum"|"avg"|...)`，给传播规则提供 reduce 语义提示（见下文 `_common_rules.einop_rule`）。

### 2.4 `_StridedShard` ↔ GSPMD nested-strided 的实现细节

[`placement_types.py:367-599`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/placement_types.py) 的 `_StridedShard(dim, split_factor)` 是为 **FSDP+TP 这种"先 TP 后 FSDP"右到左分片**专门加的子类：

- 默认的 `Shard` 按 `placements` 顺序**从左到右**依次切分。
- 当用户用 `fully_shard` 把已经按 TP 切过的张量再按 FSDP 切时，物理顺序变成右到左。`_StridedShard` 携带 `split_factor`，让 redistribute 知道"在我之前已经切了几份"，从而在做 `all_gather` 后能正确 reorder（注释里那段 "01324# -> 01234" 的例子就是 reorder 算法）。

这是一个**比 GSPMD 论文更细致的扩展**，GSPMD 的 nested sharding 用 `Offset` 兼容 shard 处理，PyTorch 用 `split_factor` 直接锁定了顺序。

---

## 3. `DTensorSpec`：GSPMD `dims_mapping` 的等价物

**GSPMD（论文 §3.1）**：`mesh_split(t, mesh, dims_mapping)` 中 `dims_mapping` 是一个长度等于 `t.ndim` 的列表，每个元素 `-1` 表示 replicate、`j>=0` 表示 shard 在 mesh 维 `j`。

**DTensor 实现**：[`_dtensor_spec.py:25`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/_dtensor_spec.py) 的 `DTensorSpec` 把 `placements: tuple[Placement, ...]` 和 `mesh: DeviceMesh` 绑定在一起，并通过 `dim_map` property 转成 GSPMD 风格：

```python
@property
def dim_map(self) -> list[int]:
    """dim_map[i] = j 表示 tensor dim i shard 在 mesh dim j；-1 表示 replicate."""
    r = [-1] * self.ndim
    for i, placement in enumerate(self.placements):
        if placement.is_shard():
            shard_dim = cast(Shard, placement).dim
            assert r[shard_dim] == -1, "不支持 [Shard(0), Shard(0)] 这类 hybrid sharding"
            r[shard_dim] = i
    return r
```

- **GSPMD 支持同 tensor dim shard 到多 mesh dim**（也就是 nested sharding），DTensor 当前的 `dim_map` 在创建时**不允许**（assert）；要表达这种语义，需要用 `Partial` + `Shard` 的组合，或者用上面的 `_StridedShard`。
- `from_dim_map(mesh, dim_map, sums, tensor_meta)` 是反向构造工具，传播规则内部用得多。
- `num_shards_map` 记录每个 tensor 维有多少 shard，用于 FSDP 这类"沿同一 dim 嵌套 shard"的合法性检查。
- `tensor_meta`（shape / stride / dtype）是传播阶段由 fake-tensor 推断出来的（`_sharding_prop.py:153` 的 `_propagate_tensor_meta_non_cached` 用 `FakeTensorMode()` 跑一遍 op 拿到）。

---

## 4. ShardingPropagator：GSPMD §3.5 的 PyTorch 实现

**GSPMD 论文 §3.5 "Intuitive sharding completion"**：用户只在少数张量上写注解，编译器按算子规则把注解沿图传播，并在不兼容时插入通信。

**DTensor 实现**：[`_sharding_prop.py:52`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/_sharding_prop.py) 的 `ShardingPropagator`：

```python
class ShardingPropagator:
    def __init__(self):
        self.op_strategy_funcs: dict[OpOverload, Callable] = {}      # 用于有多种合法策略的 op
        self.op_to_rules: dict[OpOverload, Callable] = {}            # 用于唯一映射的 op
        self.propagate_op_sharding = LocalLRUCache(self.propagate_op_sharding_non_cached)
```

注册方式（[`_sharding_prop.py:98-152`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/_sharding_prop.py)）：

- `register_op_strategy(op, func)`：让 op 返回 `OpStrategy`——一个**所有合法 `OpSpec`** 的列表（含每种方案的 `redistribute_cost`）。`ShardingPropagator` 会选**代价最小**或**不需要 redistribute** 的那个（`_select_strategy`，`[_sharding_prop.py:558`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/_sharding_prop.py)）。
- `register_sharding_prop_rule(op, func)`：让 op 给定输入 spec 后**唯一决定**输出 spec；如果遇到不支持的输入，返回 `output_spec=None` 并附 `redistribute_schema`（即"resample to this spec 后再试一次"）。

### 4.1 与 GSPMD 的关键差异：策略搜索 vs. 唯一映射

| 维度 | GSPMD | DTensor |
|---|---|---|
| 输入解析 | 用户显式写注解 | 用户写注解 OR 调用方（如 `parallelize_module`）自动注入 |
| 输出分片计算 | 按算子规则**唯一**确定，必要时插入通信 | 枚举**所有**合法 `OpSpec`，按代价挑一个（必要时插入 reshard） |
| Partial 处理 | 输出端隐式 partial sum | `Partial` 是第一类 placement，可显式保留 |
| 循环传播 | 多轮迭代直到不动点 | 单次推理 + 必要时自动 reshard（下一轮 op 时再传播） |

简单说，DTensor 把 GSPMD 的"循环迭代传播"近似成"单次传播 + 自动 redistribute"，靠 LRU cache（`propagate_op_sharding`）避免重复计算。

### 4.2 传播入口：`propagate(op_info)`

[`_sharding_prop.py:318-329`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/_sharding_prop.py)：

```python
def propagate(self, op_info: OpInfo) -> None:
    # 调用 op_info.schema（OpSchema）对应的策略函数
    output_sharding = self.propagate_op_sharding(op_info.schema)
    op_info.output_sharding = output_sharding
```

`OutputSharding` 包含：
- `output_spec`：输出的 `DTensorSpec`（含 placements）；
- `redistribute_schema`：当输入分片不在合法集合里时，**建议把输入转成什么分片**；
- `needs_redistribute`：是否需要先把输入 reshard。

这正好对应 GSPMD 论文的"preserved dimensions in operators" / "merging compatible shardings" / "insert communication when incompatible" 三条规则。

---

## 5. Per-op 规则：`einop_rule` = GSPMD "Merging compatible shardings"

**GSPMD 论文 §3.5 + Figure 3**：Dot 算子根据输入分片做兼容性检查，若两条 shardings compatible 就合并成更细的 tiled，否则插入通信。

**DTensor 实现**：[`_ops/_common_rules.py:42`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/_ops/_common_rules.py) 的 `einop_rule(equation, ...)`——把任意 Einsum 风格的算子统一用 `equation` 描述：

```python
def einop_rule(equation, op_schema, *, linearity=False, enforce_sharding=None):
    # 解析 equation，如 "mk,kn->mn"
    inputs, outputs = equation.split("->")
    input_dims, output_dims = inputs.split(","), outputs.split(",")

    dim_to_sharding: dict[str, int] = {}    # 维度字符 -> mesh dim
    pending_sums_counter: dict[int, int] = {}  # 哪些 mesh dim 有未归约的 partial sum
    seen_shardings: dict[int, str] = {}    # mesh dim -> 已用过的 dim 字符
    needs_reshard = False

    def merge_sharding(dim, a, b):
        # a 和 b 兼容（同 mesh dim 或都是 -1）就直接取；
        # 否则设为需要 reshard（让 replicate 那一侧去跟 shard 对齐）。
```

逐 op 维的规则（[`_einsum_strategy.py:85`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/_ops/_einsum_strategy.py) 的注释把这套规则讲得很清楚）：

| 情况 | 输入 | 输出 |
|---|---|---|
| 两输入都在 contracting dim 上 shard 同 mesh 维 | `S(c)`, `S(c)` | `Partial()` |
| 两输入都在 batch dim 上 shard | `S(b), S(b)` | `S(b)` |
| 一输入在 free dim 上 shard | `S(f), R` 或 `R, S(f)` | `S(f)` |
| Linear / general Einsum | 上述的笛卡尔积 | 上述的笛卡尔积 |

这正是 GSPMD 论文 Figure 3 的 `Dot` 算子分片传播图所表达的意思。

### 5.1 `mm` / `bmm` 的策略生成

[`_matrix_ops.py:215-242`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/_ops/_matrix_ops.py)：

```python
@register_op_strategy(aten.mm.default)
def mm_strategy(op_schema):
    return _mm_like_strategy("mk,kn->mn", mesh, op_schema)

@register_op_strategy(aten.bmm.default)
def bmm_strategy(op_schema):
    return _mm_like_strategy("bmk,bkn->bmn", mesh, op_schema)
```

`gen_einsum_strategies`（[`_einsum_strategy.py:85`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/_ops/_einsum_strategy.py)）枚举所有合法 `(input_specs, output_spec, redistribute_cost)`，再由 `_select_strategy` 选**最小通信代价**的那个。这相当于把 GSPMD §3.5 的"compatible shardings 合并"和"incompatible 时插入通信"合并到一个**搜索 + 代价评估**框架里。

### 5.2 Pointwise 规则 = GSPMD "Preserved dimensions"

[`_common_rules.py:225`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/_ops/_common_rules.py) 的 `pointwise_rule` 把任意 elementwise op 转成 einsum equation：

```python
def pointwise_rule(op_schema, linearity=False):
    # 根据各输入 ndim 拼出形如 "ij,ij->ij" 的 equation
    fmt = f"{','.join(p for p in dimchars)}->{out_dimchars}"
    return einop_rule(fmt, op_schema, linearity=linearity, ...)
```

这就是 GSPMD 论文里"elementwise 算子保留输入输出分片"的统一抽象——`ReLU`、`LayerNorm`、binary ops 等都走这条路。

### 5.3 `linearity` 与 `Partial` 传播

**GSPMD 论文没显式讨论这点**，但 `einop_rule` 的 `linearity=True` 是 DTensor 工程上的一个重要能力：

- 如果 `f(a + b) = f(a) + f(b)` 成立（如 `add`、`matmul`），就可以让多个 `Partial` 在 mesh 维上**累积**——避免中途做 AllReduce 还原成 Replicate。
- DTensor 把 `add` / `addmm` 等注册为 `linearity=True`，因此 `S(0) @ S(1)` 输出 `Partial` 后还能跟别的 `Partial` 张量相加，最后只在必要时 reduce 一次（对应 GSPMD 论文 Figure 7 的 `RS`/`AR` 选择）。
- 这条性质只有 sum reduction 适用；min/max/prod 因为非线性会被传播规则丢弃 `Partial`。

---

## 6. Resharding：GSPMD §4.2 的通信原语落地

**GSPMD 论文 §4.2**：当源/目标分片注解不一致时，编译器自动插入 `AllReduce / AllGather / ReduceScatter / AllToAll / CollectivePermute`。

**DTensor 实现**：[`_redistribute.py`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/_redistribute.py)：

### 6.1 `_gen_transform_infos`：把多维 reshard 拆成单 mesh 维的串行变换

[`_redistribute.py:31-146`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/_redistribute.py)：

```python
def _gen_transform_infos_non_cached(src_spec, dst_spec):
    # 1. 先计算每个 mesh 维上的 logical shape（处理 uneven sharding）
    mesh_dims_to_logical_shape = [initial_logical_shape]
    for i, src in enumerate(src_spec.placements):
        if isinstance(src, Shard) and i < device_mesh.ndim - 1:
            local_shard_size, _ = src._local_shard_size_and_offset(...)
            new_logical_shape = list(current_logical_shape)
            new_logical_shape[src.dim] = local_shard_size
            mesh_dims_to_logical_shape.append(new_logical_shape)
        else:
            mesh_dims_to_logical_shape.append(current_logical_shape)

    # 2. 反向遍历 mesh 维，检测嵌套 shard 是否对齐
    #    例如 (S(0), S(0)) -> (R, S(0))，第一个 S(0) 需要先 AllGather 成 R
    for mesh_dim in reversed(range(len(current_placements))):
        ...

    # 3. 正向遍历，补齐剩余变化
    for mesh_dim, (current, target) in enumerate(...):
        if current != target:
            transform_infos.append(...)
```

这对应 GSPMD §3.5 "merging compatible shardings" 中的"先识别嵌套 shard，对齐后展开"逻辑。

### 6.2 `redistribute_local_tensor`：每对 (current, target) placement 选一个原语

[`_redistribute.py:157-278`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/_redistribute.py) 的核心是 case 分支：

| 源 → 目标 | 实现 | 对应 GSPMD 原语 |
|---|---|---|
| `Partial → Replicate` | `partial._reduce_value()` → `funcol.all_reduce` | `AllReduce` |
| `Shard → Replicate` | `shard._to_replicate_tensor()` → `funcol.all_gather_tensor` | `AllGather` |
| `Partial → Shard` | `partial._reduce_shard_value()` → `reduce_scatter_tensor` | `ReduceScatter` |
| `Replicate → Shard` | `shard._replicate_to_shard()` → 本地 `chunk + clone` | 无通信 |
| `Shard(dim_a) → Shard(dim_b)` | `shard._to_new_shard_dim()` → `shard_dim_alltoall` | `AllToAll` |
| `Replicate → Partial(sum)` | `partial._partition_value()` → `tensor / num_chunks` | 纯本地除法 |

注意 `Replicate → Partial(sum)` 是**纯本地除法**（用 mesh 维大小分摊），没有通信——这是 DTensor 把"replicate 转成 partial"做成廉价操作的关键点。

### 6.3 `Redistribute` autograd Function

[`_redistribute.py:281-401`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/_redistribute.py)：

```python
class Redistribute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, device_mesh, placements, ...):
        ...
        if current_spec.placements != placements:
            target_spec = DTensorSpec(device_mesh, placements, ...)
            output = redistribute_local_tensor(local_tensor, current_spec, target_spec, ...)
        ...

    @staticmethod
    def backward(ctx, grad_output):
        # 反向 reshard 回原来的 placement
        # 但有一个特殊处理：如果原 spec 是 Partial，则 backward 时把它"提升"为 Replicate
        # ——这是为了避免不必要的 all-reduce
```

`is_backward=True` 的特殊路径对应 GSPMD 论文没有显式讨论但工程上很关键的**反向梯度重分片优化**：当 reshard 是 `Partial → Replicate` 时，前向多花的 `all_reduce` 在反向里会被"撤销成 `Replicate` 直接走"——避免梯度累积阶段做无意义的 reduce。

### 6.4 不直接支持：跨 mesh 通信

```python
if current_spec.mesh != target_spec.mesh:
    # TODO: alltoall/permute reshuffling to change device_mesh if they are not the same
    raise NotImplementedError("Cross device mesh comm not supported yet!")
```

DTensor 当前不支持跨 mesh 的 reshard（要重建 mesh 或 all-to-all permutation）。GSPMD 论文 §3.3 里也只是建议借助 wrapper 库处理这种场景。

---

## 7. `_conv_ops.py`：GSPMD §4.3 Halo exchange 的 PyTorch 版本

**GSPMD 论文 §4.3**：Conv、Pooling、Reverse 等算子需要邻域数据，必须在 padding/slice 前做 halo exchange。

**DTensor 实现**：[`_ops/_conv_ops.py`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/_ops/_conv_ops.py) 通过对 `aten.convolution.default` 注册 `OpStrategy`，枚举在哪些 mesh 维上能 shard input / weight：

- 对 `Shard(N)` 输入（N=batch 或 spatial dim），在 spatial 维 sharding 时需要做**邻域 all-gather**（对应 halo exchange）。
- 对 `Shard(C_in)`（输入通道）或 `Shard(C_out)`（输出通道），通过 1×1 卷积或专门算子避免 halo exchange。
- uneven padding 通过 `compute_local_shape_and_global_offset` 算出 local shape，再 pad。

这部分代码实现非常细致，对应 GSPMD 论文 Figure 5 的 halo exchange 例子。

---

## 8. `parallel/`：用户面对的 TP / DP / FSDP 接口 = GSPMD 的"high-level parallel styles"

GSPMD 的核心是注解 + SPMD 编译，但用户实际写模型时希望直接拿到 ColwiseParallel / RowwiseParallel / FSDP 这种"语义化"接口。DTensor 的 [`parallel/`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/parallel/) 子包就是这一层。

### 8.1 `ColwiseParallel` / `RowwiseParallel` = GSPMD 论文 §5.1 的 2D sharding

[`parallel/style.py:45-322`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/parallel/style.py)：

```python
class ColwiseParallel(ParallelStyle):
    def __init__(self, *, input_layouts=None, output_layouts=None, ...):
        # input: Replicate()，output: Shard(-1)
        # 对应 GSPMD 的 WQ/WK/WV 按列切分

class RowwiseParallel(ParallelStyle):
    # input: Shard(-1)，output: Partial()（因为 a @ b 在 reduce 维上需要 all-reduce）
    # 对应 GSPMD 论文 Wo 的 row-wise 切分
```

这正好是 GSPMD 论文 §5.1 Megatron-style TP 的注解——`Colwise` 是 output 维切分，`Rowwise` 是 contracting 维切分 + 输出 `Partial`。`SequenceParallel` 把 LayerNorm 输入沿 sequence 维切，对应 GSPMD §5.2 的细粒度分片。

### 8.2 FSDP = GSPMD 论文 §5.1 的 weight-update sharding

[`parallel/fsdp.py`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/parallel/fsdp.py) 把 weight + gradient + optimizer state 都按 mesh dim 切：

- 前向：`weight` 是 `Shard(0)`，输入是 `Replicate`。
- 反向：触发 `ReduceScatter` 把梯度切分到各 rank。
- 优化器：每个 rank 只更新自己那份 weight。

对应 GSPMD 论文 Figure 7 的 `RS` (ReduceScatter) 通信。

### 8.3 `ddp.py` = GSPMD 论文 §2.1 的纯 data parallel

[`parallel/ddp.py`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/distributed/tensor/parallel/ddp.py) 简单地做 `input Replicate`，前向 no-op，反向触发一次 AllReduce。

---

## 9. 与 GSPMD 论文相比，DTensor 做了哪些工程取舍

| 维度 | GSPMD 论文做法 | DTensor 取舍 |
|---|---|---|
| 分片传播 | 迭代到不动点（多轮） | 单次传播 + 自动 reshard，LRU cache 复用 |
| Partial 处理 | 隐式存在于输出端 | 显式第一类 placement，可保留到 reduce 前 |
| 算子策略 | 唯一确定 + 必要时插入通信 | 枚举所有合法策略 + 选最小代价 |
| 嵌套并行 | 任意 nested tiled 通过 Offset 兼容 | 不允许 dim_map 重复，需要用 `Partial`+`Shard` 组合或 `_StridedShard` |
| 流水线 | §3.3 把 PP 降成张量分片 | DTensor 不直接做 PP（由 `PiPPy` / `torchgpipe` 处理） |
| 编译 | XLA SPMD partitioner | eager mode + `torch.compile`（DTensor 配合 Dynamo） |
| 跨 mesh 通信 | wrapper 库自定义 | 当前抛 `NotImplementedError` |
| 静态 shape | padding + DynamicSlice | `Shard._split_tensor` 用 padding，`compute_local_shape_and_global_offset` 算 local shape |
| 非均匀分片 | 支持但需要小心 | 一等支持（`_local_shard_size_and_offset`），只是 experimental |
| Pipeline 调度 | GSPMD wrapper 库自己实现 | 由 PyTorch 生态的 PP 库实现，DTensor 只管张量分片 |

---

## 10. 阅读代码的推荐路径

1. **入口**：`device_mesh.py` → 看 `__init__`、`_init_process_groups`、`__getitem__`。
2. **基本语义**：`placement_types.py` → `Shard._shard_tensor` / `Replicate._replicate_tensor` / `Partial._reduce_value`，理解每个原语在底层调什么 funcol。
3. **Spec 表示**：`_dtensor_spec.py` → `dim_map`、`num_shards_map`、`from_dim_map`。
4. **传播机制**：`_sharding_prop.py` → `propagate_op_sharding_non_cached`、`_select_strategy`、`_adjust_shape_and_stride_args`。
5. **通用规则**：`_ops/_common_rules.py` → `einop_rule`、`pointwise_rule`。
6. **矩阵算子**：`_ops/_matrix_ops.py` + `_ops/_einsum_strategy.py` → `mm`、`bmm`、`sdpa` 的策略生成。
7. **Reshard**：`redistribute.py` → `_gen_transform_infos_non_cached`、`redistribute_local_tensor` 的 case 表。
8. **卷积 halo**：`_ops/_conv_ops.py` → 邻域 allgather 的处理。
9. **用户层 API**：`parallel/style.py`、`parallel/api.py` → `parallelize_module`、`ColwiseParallel` / `RowwiseParallel` / `SequenceParallel`。
10. **FSDP 集成**：`parallel/fsdp.py` + `placement_types._StridedShard` → 理解为什么需要 `_StridedShard`。

---

## 11. 关键 takeaways

1. **DTensor 是 GSPMD 的"Pytorch 化重写"**：抽象（Placement + DeviceMesh + DTensorSpec）几乎一一对应，但工程上做了很多 PyTorch 化的取舍（eager mode、torch.compile 兼容、NCCL 友好）。
2. **`Partial` 是 DTensor 比 GSPMD 更显式的关键**：通过把"partial sum"提升为 placement，让线性算子（mm/add）能延迟通信，是通信优化的重要杠杆。
3. **策略搜索 vs. 规则传播**：GSPMD 是"规则 + 通信插入"的纯声明式方法；DTensor 把这变成"枚举 + 选最小代价"，更符合 PyTorch eager 用户的直觉，但需要为每个 op 写一份 `OpStrategy`。
4. **`_StridedShard` 揭示了 GSPMD nested sharding 的真实复杂度**：物理上 TP-then-FSDP 与 FSDP-then-TP 是不同语义，DTensor 必须显式区分；GSPMD 用 Offset 来兼容，PyTorch 选择显式建模。
5. **Resharding 是 DTensor 的"通信插入器"**：`_redistribute.py` 的 case 表几乎是 GSPMD 论文 §4.2 的 Python 版实现，反向传播的 `Partial → Replicate` 优化是 PyTorch 工程侧的额外考量。

如果要继续深入，建议做下面两件事：

- 跑一次 `torch.distributed.tensor` 的单测，对照 `_sharding_prop` 在 `aten.mm`、`aten.add`、`aten.layer_norm` 上的实际行为；
- 拿一个简单模型（如 MLP）用 `parallelize_module` 切到 TP + FSDP mesh 上，跑 `torch.compile`，看 Dynamo 如何把 DTensor 翻译成 XLA/inductor。
