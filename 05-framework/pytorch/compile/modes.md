# torch.compile 的 mode 详解

> 调研范围：PyTorch 2.9.1 源码 `torch._inductor`、`torch._inductor.config`、`torch._inductor.runtime.triton_heuristics`
> 目标：把 `torch.compile(mode=...)` 的 4 种 mode 拆开讲透，从配置开关到内部机制

`torch.compile(fn, mode=...)` 接受 4 种 mode，每种对应一组 **Inductor config 开关**。这些开关决定了：要不要开 **CUDA Graph**、要不要做 **max-autotune**、要不要做 **coordinate descent**。

---

## 0. 一张总览表

源码出处 [`torch/_inductor/__init__.py:332`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_inductor/__init__.py)：

```python
mode_options: dict[str, dict[str, bool]] = {
    "default":                 {},
    "reduce-overhead":         {"triton.cudagraphs":            True},
    "max-autotune-no-cudagraphs": {
        "max_autotune":              True,
        "coordinate_descent_tuning": True,
    },
    "max-autotune": {
        "max_autotune":              True,
        "triton.cudagraphs":         True,
        "coordinate_descent_tuning": True,
    },
}
```

| mode | cudagraphs | max-autotune | coordinate descent | 适用 |
|---|---|---|---|---|
| `default` |  | ✗ | ✗ | 通用；推荐作为起点 |
| `reduce-overhead` | ✓ | ✗ |  | launch overhead 占主导（小 kernel 多、batch=1） |
| `max-autotune-no-cudagraphs` |  | ✓ | ✓ | 形状变化剧烈、要 autotune 但 cudagraph 不划算 |
| `max-autotune` | ✓ | ✓ | ✓ | 计算密集、有大 GEMM/Conv，能接受长编译时间 |

> 注：这是 PyTorch 2.9 当前的官方定义。早期版本（如 2.0）的 `reduce-overhead` 还包括 `triton.unique_kernel_names` 等开关，已被合并/移除。

下面按"配置开关的内部机制"展开。

---

## 1. 共有底层：不管哪个 mode 都做的事

不论你选哪个 mode，Inductor 都至少做以下事情：

1. **算子融合（fusion）**：
   - elementwise 链 → 一个 Triton kernel；
   - reduction + 后续 elementwise → reduction kernel + epilogue kernel；
   - GEMM + epilogue（bias add / ReLU）→ Triton GEMM template（`aten.mm` 默认走 `aten`，不调 Triton）。
2. **buffer 复用（memory planning）**：[`torch/_inductor/memory_planning.py`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_inductor/memory_planning.py) 算每个 intermediate tensor 的生命周期，把不同生命周期的 buffer 复用同一段显存。
3. **常量折叠（constant folding）**：[`torch/_inductor/constant_folding.py`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_inductor/constant_folding.py) 在 trace 阶段把可静态求值的子图直接算出。
4. **每个 Triton kernel 自带 autotune**：每个 `triton.jit` kernel 在 `runtime/triton_heuristics.py` 里有一个 `TritonConfig` 候选表（num_warps、num_stages、BLOCK_SIZE），第一次跑会逐个 benchmark 选最快的。

也就是说，**`default` 不是"什么都不做"**，它已经做了 fusion + memory planning + 内置 kernel autotune。后面 3 个 mode 是叠加额外优化。

---

## 2. `default`：基线

### 2.1 配置

无额外开关，纯基础流水线。

### 2.2 内部机制

- **Fusion**：`scheduler` 把 FX graph 转成 fusion group，每个 group 对应一个 Triton kernel（elementwise）或 Triton template（reduction / matmul）。
- **GEMM 走 ATen**：`aten.mm` / `aten.addmm` 默认不调 Triton GEMM，直接 dispatch 到 `torch.matmul`（cuBLAS）。Inductor 只是在它外面做 epilogue 融合。
- **每个 Triton kernel 内部 autotune**：通过 `@triton.autotune` 装饰器（`triton_heuristics.py` 里 `Autotuner`），对每个 kernel 在 1~N 个 config 之间做 `do_bench` 选最快的。
- **不捕获 cudagraph**：每次调用都要走 Python → ATen → cuLaunchKernel 的 launch 路径。

### 2.3 性能特征

- **编译时间**：每个 kernel 几秒（取决于 kernel 数量）。
- **运行时间**：相比 eager 通常 1.3x–1.8x，主要来自 fusion 减少 kernel launch 和内存往返。
- **优势场景**：常规训练/inference，没有极端 launch overhead 问题。

### 2.4 例子

```python
@torch.compile
def f(x, w):
    return torch.relu(x @ w)

f(torch.randn(64, 128), torch.randn(128, 64))
```

---

## 3. `reduce-overhead`：消除 Python launch overhead

### 3.1 配置

```python
torch._inductor.config.triton.cudagraphs = True
```

源码 [`torch/_inductor/config.py:1221`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_inductor/config.py)：

```python
# Use cudagraphs on output code
cudagraphs = os.environ.get("TORCHINDUCTOR_CUDAGRAPHS") == "1"
# Use cudagraph trees for memory pooling if `cudagraphs` is True
cudagraph_trees = True
```

### 3.2 内部机制：cudagraphify

源码 [`torch/_inductor/compile_fx.py:1733`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_inductor/compile_fx.py) 的 `cudagraphify`：

1. **第一次调用**：进入一个独立 stream，开始 **cudaStreamBeginCapture**。所有后续的 CUDA kernel launch 都被记录而不是真正执行。
2. **capture 结束**：调 cudaGraphInstantiate，得到一个可执行的 `cudaGraph_t`。
3. **后续调用**：直接 `cudaGraphLaunch`，跳过 Python、跳过 cuLaunchKernel，单次 launch overhead 从 ~10µs 降到 ~1µs。

关键配置（`config.py`）：

```python
cudagraph_trees = True                  # 启用 cudagraph trees
cudagraph_skip_dynamic_graphs = False   # dynamic shape 时是否跳过 cudagraph
cudagraph_capture_sizes = None          # 显式指定要 capture 的 shape
cudagraph_support_input_mutation = True
cudagraph_unexpected_rerecord_limit = 128
cudagraph_dynamic_shape_warn_limit = 50
force_cudagraphs_warmup = False
```

### 3.3 cudagraph_trees：为什么需要它

Naive cudagraph 实现有个问题：**每次 shape 变化都要重新 capture**。`cudagraph_trees`（[`torch/_inductor/cudagraph_trees.py`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_inductor/cudagraph_trees.py)）用一棵树按 shape 复用：

```
root
├── shape A (M=512)
│     ├── static_a_0  (cudagraph 实例)
│     └── static_a_1  (另一组 static input)
├── shape B (M=1024)
│     └── static_b_0
└── shape C (M=128)
      └── static_c_0
```

每个 shape 第一次走 capture，之后 replay。`cudagraph_unexpected_rerecord_limit = 128` 是"同一 shape 下允许重 capture 的上限"，超过就警告（多半是输入 tensor 地址变了）。

### 3.4 cudagraph 的代价

- **第一次调用**会做 capture，慢很多；
- **Input mutation**：如果输入 tensor 被 in-place 改了，cudagraph 必须重 capture；
- **Dynamic shape 频繁变化**：每变一次都 capture，反而比不 capture 更慢；
- **cudagraph-unsafe op**（CPU/GPU 同步、随机数生成、`cudaMemcpy`）会拒绝 capture（`cudagraph_or_error=False` 时静默 fallback 到无 cudagraph 路径）。

### 3.5 性能特征

- **编译时间**：跟 default 几乎一样；
- **运行时间**：相比 default 又快 10%~30%，主要来自 launch overhead 消除；
- **优势场景**：小 batch、transformer decoder（attention 拆出很多小 kernel）、LLM inference（batch=1, prompt 短）。

### 3.6 例子

```python
@torch.compile(mode="reduce-overhead")
def step(x):
    return torch.nn.functional.gelu(x @ w.T)

# 第一次：capture；之后：replay
for _ in range(100):
    step(torch.randn(8, 64))
```

开 log 验证：

```python
import torch._logging
torch._logging.set_logs(cudagraphs=True)
step(torch.randn(8, 64))
```

会看到 `cuGraphCreate` / `cuGraphInstantiate` / `cuGraphLaunch` 的日志。

---

## 4. `max-autotune-no-cudagraphs`：autotune 但不 cudagraph

### 4.1 配置

```python
max_autotune               = True
coordinate_descent_tuning  = True
# cudagraphs = False
```

源码 [`config.py:440, 552`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_inductor/config.py)：

```python
max_autotune = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE") == "1"
coordinate_descent_tuning = (
    os.environ.get("TORCHINDUCTOR_COORDINATE_DESCENT_TUNING") == "1"
)
```

### 4.2 `max_autotune` 打开什么

`max_autotune` 不只是一个开关，它会让 Inductor **多走几遍不同的 IR→kernel 路径并 benchmark**：

#### 4.2.1 GEMM 后端 autotune

源码 [`config.py:498`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_inductor/config.py)：

```python
# 控制 GEMM autotune 的 backend 候选
max_autotune_gemm_backends = os.environ.get(
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN,TRITON,CPP"
)
# 控制 conv autotune 的 backend
max_autotune_conv_backends = os.environ.get(
    "TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS", "ATEN,TRITON"
)
```

每个 `aten.mm` 会生成 3 个候选实现：

| backend | 实现 |
|---|---|
| `ATEN` | 直接调 `torch.matmul`（cuBLAS） |
| `TRITON` | Triton GEMM template（`aten.mm` 降级成 `tl.dot`） |
| `CPP` | C++/CUTLASS GEMM（带 autotune） |

每个 candidate 跑 `do_bench` 测时，选最快的。

#### 4.2.2 GEMM search space

```python
# config.py:514
max_autotune_gemm_search_space: Literal["DEFAULT", "EXHAUSTIVE"] = os.environ.get(
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE", "DEFAULT"
)
```

- `DEFAULT`：~10 个 Triton matmul config（BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages）；
- `EXHAUSTIVE`：~200+ 个 config，覆盖 split-K、L2 swizzle、各种 stage 数。

#### 4.2.3 FlexAttention search space

```python
# config.py:521
max_autotune_flex_search_space: Literal["DEFAULT", "EXHAUSTIVE"] = ...
```

控制 FlexAttention 中 BLOCK_M / BLOCK_N / warps / stages 的搜索空间。

#### 4.2.4 benchmark_epilogue_fusion

源码 [`config.py:656`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_inductor/config.py)：

```python
# For Triton Templates, select fastest of best template + epilogue vs best template + separate epilogue kernel
benchmark_epilogue_fusion = (
    os.environ.get("TORCHINDUCTOR_BENCHMARK_EPILOGUE_FUSION", "0") == "1"
)
```

会测：把 epilogue（bias + ReLU）合到 Triton kernel vs 单独跑一个 epilogue kernel，哪个更快。

#### 4.2.5 其它 max_autotune 子开关

```python
max_autotune_pointwise         # 强制 pointwise kernel 走 autotune（默认 false）
max_autotune_gemm              # 强制 GEMM 走 autotune
max_autotune_subproc_result_timeout_seconds = 60.0  # autotune 子进程超时
```

### 4.3 `coordinate_descent_tuning` 打开什么

源码 [`runtime/triton_heuristics.py:1125`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_inductor/runtime/triton_heuristics.py)：

```python
def coordinate_descent_tuning(self, launcher, *args, **kwargs):
    """
    Coordinate descent tuning can be run with or without max-autotune.

    The only difference between these two is the starting config for
    coordinate_descent tuning. E.g., assuming regular autotune only get one
    config C1; while max-autotune get 4 configs C1, C2, C3, C4 and
    max-autotune figure out C3 is the best.

    Then if coordinate descent tuning is run with max-autotune disabled,
    it will start from C1;
    while if coordinate descent tuning is run with max-autotune enabled,
    it will start from C3.
    """
```

具体算法（`coordinate_descent_tuner.py` 的 `CoordescTuner`）：

1. 拿当前 launcher 的 config（BLOCK_M / BLOCK_N / BLOCK_K / num_warps / num_stages）做起点；
2. 在每个维度上**单独**±1 step（比如 num_warps 从 4 试到 8，再试到 2）；
3. 哪个方向更快，就走那个方向，直到所有维度都"小步改了也没改进"；
4. 返回最优 config。

文档字符串里那句"C3 是 max-autotune 选出的最优起点"是关键：max-autotune 已经把搜索空间砍过一遍，coordinate descent 在此基础上做精细调优，所以两者**通常需要一起开**。

#### 4.3.1 相关子配置

```python
coordinate_descent_check_all_directions = False  # True: 检查 ±2 个方向；False: 只检查 +1
coordinate_descent_search_radius       = 1      # 单方向最大步数
```

### 4.4 性能特征

- **编译时间**：相比 default **长一个数量级**。每个 GEMM/Conv 要 autotune 几个 backend，每个 Triton kernel 要做 coordinate descent，单图可能几分钟到几十分钟。
- **运行时间**：相比 default 又快 1.2x–2x，来源：
  - GEMM 切到 Triton + autotune 通常比 cuBLAS 快（因为可以做 epilogue fusion，省一次读 + write）；
  - Triton kernel BLOCK 参数被 coordinate descent 精调过。
- **优势场景**：训练 / 推理稳定运行，shape 变化不大（变一次要重新 autotune）。

### 4.5 例子

```python
@torch.compile(mode="max-autotune-no-cudagraphs")
def step(x, w):
    h = x @ w           # 触发 GEMM autotune
    return h.relu()
```

开 log：

```python
torch._inductor.config.max_autotune_report_choices_stats = True
torch._logging.set_logs(benchmarking=True)
```

会看到每个算子候选实现的 benchmark 表格。

---

## 5. `max-autotune`：把上面全部叠起来

### 5.1 配置

```python
max_autotune              = True
triton.cudagraphs         = True
coordinate_descent_tuning = True
```

跟 `max-autotune-no-cudagraphs` 唯一的区别就是 `triton.cudagraphs = True`。

### 5.2 跟 `max-autotune-no-cudagraphs` 的取舍

| 维度 | max-autotune-no-cudagraphs | max-autotune |
|---|---|---|
| GEMM/Conv autotune | ✓ | ✓ |
| Coordinate descent | ✓ | ✓ |
| cudagraph | ✗ | ✓ |
| 形状变化友好度 | **友好**（不重 capture） | **不友好**（每次新 shape 都要重 capture） |
| Launch overhead | 仍有 | 几乎消除 |
| 典型场景 | 训练 / 形状变化推理 | 推理服务（shape 稳定）、单步 forward |

### 5.3 典型用法

```python
# 推理服务，shape 完全固定
@torch.compile(mode="max-autotune")
class Infer:
    def forward(self, x): ...
```

如果是训练：

- **第 1 步编译**：可能 5–30 分钟（取决于 GEMM 数量）；
- **第 2 步之后**：每个新 batch shape 会重新触发 cudagraph capture + autotune cache miss，要几百 ms ~ 几秒；
- **稳定后**：每个 step 接近理论最优。

---

## 6. 4 种 mode 的 launch 路径对比

```
default:
   Python fn → compiled_fn (Python wrapper)
                       │
                       └─ Triton kernel @ cuLaunchKernel   ← 每个 kernel ~10µs launch

reduce-overhead:
   Python fn → compiled_fn
                       │
                       └─ cudaGraphLaunch (one shot)         ← ~1µs / launch

max-autotune-no-cudagraphs:
   Python fn → compiled_fn (autotuned)
                       │
                       └─ Triton / ATen / CUTLASS @ cuLaunchKernel

max-autotune:
   Python fn → compiled_fn (autotuned)
                       │
                       └─ cudaGraphLaunch (one shot, with autotuned kernels)
```

---

## 7. 几个常见误区

### 7.1 `max-autotune` 不一定更快

GEMM autotune 有时候会选到**比 cuBLAS 慢**的 Triton 实现（尤其在 RTX 3090 / A100 上 cuBLAS 已经很好）。`max_autotune_gemm_backends` 设成 `"ATEN"` 单独使用 ATen（cuBLAS）有时反而最快。

### 7.2 `reduce-overhead` 在 dynamic shape 下失效

如果你每一步的 batch size 都变，cudagraph 会一直重 capture，反而变慢。修法：
- 用 `torch._dynamo.mark_dynamic(x, 0)` 减少 recompile；
- 或者直接 `cudagraph_capture_sizes = [(b,) for b in known_shapes]` 限定 capture 范围；
- 或者退回 `default` / `max-autotune-no-cudagraphs`。

### 7.3 `coordinate_descent_tuning` 在 max-autotune 之外意义有限

不跟 max-autotune 配合时，coordinate descent 只能从一个 config 起步调；如果 max-autotune 还没帮你把候选空间砍掉，Triton 内置 autotune 的 config 选择并不好。

源码注释已经写明：
> "The only difference between these two is the starting config for coordinate_descent tuning."

### 7.4 cudagraph 跟 DDP 不太兼容

DDP 的 `allreduce` launch 是 CPU 端的 collective op，cudagraph 不能 capture。修法是 `torch._inductor.config.triton.cudagraphs = False`，或者用 `torch._dynamo.allow_in_graph(dist.all_reduce)` 把 collective 摘出去再 cudagraph capture。

---

## 8. 推荐选择路径

```
1. 先用 default，确认 fused graph 数 / kernel 数合理；
   工具：TORCH_LOGS="output_code,graph_code"

2. 看 profiler：
   - 如果 launch overhead 占比高（kernel 时间很短但 #kernels 多） → reduce-overhead
   - 如果是 matmul / conv 等大 kernel 时间占比高 → max-autotune

3. 如果 shape 变化剧烈：
   - 训练 → max-autotune-no-cudagraphs
   - 推理但 shape 经常变 → reduce-overhead（配合 mark_dynamic）

4. 最后一搏：max-autotune，能等编译时间就上。
```

---

## 9. 自定义 mode：把 mode 当成"模板"修改

`mode` 不是黑盒，可以**叠加**额外开关：

```python
import torch._inductor.config as ic

@torch.compile(mode="default", options={
    "epilogue_fusion": True,            # 几乎总是 True
    "max_autotune": True,               # 单独打开 autotune（无 cudagraph）
    "coordinate_descent_tuning": True,
    "triton.cudagraphs": False,
    "multi_kernel": 2,                  # 1: 融合大 kernel；2: 切多个小 kernel
})
def f(x): ...
```

`options` 字典里的 key 直接对应 `torch._inductor.config` 下的属性，详见 [`torch/_inductor/list_options()`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_inductor/__init__.py)（返回当前生效的所有 config 项）。

或者直接修改全局 config（影响同进程后续所有 compile）：

```python
ic.max_autotune = True
ic.coordinate_descent_tuning = True
ic.triton.cudagraphs = False
```

---

## 10. 关键源码定位速查

| 想看什么 | 文件 / 行 |
|---|---|
| 4 种 mode 的定义 | `torch/_inductor/__init__.py:332` |
| cudagraph 实现 | `torch/_inductor/compile_fx.py:1733 cudagraphify` |
| cudagraph trees | `torch/_inductor/cudagraph_trees.py` |
| cudagraph 相关 config | `torch/_inductor/config.py:1220-1267` |
| max-autotune 相关 config | `torch/_inductor/config.py:439-547` |
| coordinate descent 实现 | `torch/_inductor/runtime/coordinate_descent_tuner.py`（`CoordescTuner`） |
| coordinate descent 入口 | `torch/_inductor/runtime/triton_heuristics.py:1125 CachingAutotuner.coordinate_descent_tuning` |
| GEMM 候选实现（CUTLASS 等） | `torch/_inductor/codegen/cuda/`、`torch/_inductor/codegen/cpp_gemm_template.py` |
| benchmark 工具 | `torch/_inductor/runtime/triton_heuristics.py:884 do_bench_using_profiling` |

---

## 11. 一句话总结

| mode | 一句话 |
|---|---|
| `default` | 基础 fusion + kernel autotune |
| `reduce-overhead` | 加 cudagraph，消除 launch overhead |
| `max-autotune-no-cudagraphs` | 加 GEMM/conv 多 backend autotune + coordinate descent，shape 友好 |
| `max-autotune` | 上面全部叠起来，适合稳定 shape 的推理 |
