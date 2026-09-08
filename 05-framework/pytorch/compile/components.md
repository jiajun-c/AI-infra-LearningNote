# torch.compile 详解：从一行装饰器到 Triton 内核

> 调研范围：PyTorch 2.9.1 源码 `torch._dynamo`、`torch._inductor`、`torch._functorch`
> 目标：把 `torch.compile` 内部各个组件串起来，对照源码讲清楚"为什么这样做"

本文适合已经用过 `torch.compile(fn)` 但想搞清楚内部机制的读者。读完应该能：

1. 说清楚 `torch.compile` 从入口到出 Triton 内核的整条流水线；
2. 解释 **Dynamo / AOTAutograd / Inductor** 各自的职责；
3. 理解 **guards、graph break、recompile、dynamic shapes、cudagraphs** 的来龙去脉；
4. 知道怎么写 Triton 自定义算子接入到 `torch.compile` 中。

---

## 0. 一句话总览

`torch.compile(fn)` = **Dynamo**（字节码 → FX 图 + guards） → **AOTAutograd**（拆 joint fwd/bwd + 算子分解） → **Inductor**（FX 图 → Triton / C++ 代码） → **runtime**（`OptimizedModule` + guard 检查 + cudagraphs）。

整条流水线是**编译器栈**：Dynamo 是 trace JIT，AOTAutograd 是图重写 pass，Inductor 是真正的代码生成器。

---

## 1. 顶层架构：四层流水线

```
Python 函数 ── Dynamo ─→ FX Graph（带 guards）
                          │
                          ▼
                    AOTAutograd ─→ joint graph → 前向 / 反向两个 FX Graph（带 decompose）
                          │
                          ▼
                      Inductor ─→ Triton kernels + Python wrapper
                          │
                          ▼
                    OptimizedModule（runtime + guard check）
```

源码位置一览（基于 PyTorch 2.9.1）：

| 组件 | 主要文件 | 入口 |
|---|---|---|
| **Dynamo**（图捕获 + guards） | `torch/_dynamo/` | `eval_frame.py`、`convert_frame.py`、`symbolic_convert.py`、`output_graph.py`、`guards.py` |
| **AOTAutograd**（autograd 编译） | `torch/_functorch/` | `aot_autograd.py`、`partitioners.py` |
| **Inductor**（codegen） | `torch/_inductor/` | `compile_fx.py`、`compile_fx_inner`、`codegen/triton.py`、`codegen/cuda` |
| **运行支撑** | `torch/_dynamo/eval_frame.py` | `OptimizedModule`、`_TorchDynamoContext`、`set_eval_frame` |

---

## 2. Dynamo：字节码 → FX 图

### 2.1 入口：`torch.compile(fn)`

`torch.compile` 只是一个工厂函数，最终产物是 `OptimizedModule`：

- 入口在 `torch/_dynamo/eval_frame.py:340` 的 `OptimizedModule(torch.nn.Module)`，持有一个 `_torchdynamo_orig_callable`（原函数）和一个 guard 检查器。
- `torch.compile` 通过 `__call__` 把函数调用重定向到 Dynamo 的 frame evaluation hook。

每次 `compiled_fn(x)` 走到原 Python 字节码时，CPython 的 frame evaluation hook 会调用 Dynamo（[`eval_frame.py:139`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py)）：

```python
class DynamoStance(Enum):
    DEFAULT = "default"            # 走 trace + compile
    RUN_ONLY = "run_only"          # 只跑 eager（不 compile）
    SKIP = "skip"                  # 跳过该 frame
    DISABLE = "disable"            # 跳过该 frame + 子 frame
    FORCE_TRACE = "force_trace"    # 强制重 trace（用于投机执行）
```

`@torch.compile` 装饰器等价于 `_TorchDynamoContext()(fn)`，调用时设置 `DEFAULT` stance 并切换 `eval_frame` hook（[`eval_frame.py:590`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py)）。

### 2.2 字节码级别的 trace：CPython 帧回调

Dynamo **不依赖 `__builtins__` 的 hook**，而是利用 CPython 3.8+ 提供的 [PEP 523](https://peps.python.org/pep-0523/) 帧求值 API（`_PyEval_EvalFrameDefault` 的替代）：

```c
// torch/csrc/dynamo/eval_frame.c 中大致是：
PyObject* custom_eval_frame(PyThreadState* ts, PyFrameObject* frame, int throw) {
    return dynamo_callback(ts, frame, throw);
}
```

每当 Python 执行一个新 frame，CPython 都会调用 `custom_eval_frame`。Dynamo 在这个 hook 里：

1. 拿到 frame 对象（`types.FrameType`），拿到 code object 与字节码；
2. 进入 `convert_frame.py` 把它"翻译"成 FX 图；
3. 把翻译后的 code object **替换**回去（`transform_code_object`），后续调用直接走新字节码，绕开 hook。

这是 Dynamo 比传统 `sys.settrace` 高效的关键：它**只 trace 一次**，之后 frame 的字节码已经被替换。

### 2.3 符号化执行：`InstructionTranslator`

`torch/_dynamo/symbolic_convert.py:1150` 的 `InstructionTranslatorBase` 是核心：

- 维护一个 **Python 栈**（`Stack`）、**变量跟踪器集合**（`VariableTracker`，`VT`）、**FX 图构建器**（`OutputGraph`）；
- 对每个 Python bytecode 指令（`LOAD_FAST`、`CALL_FUNCTION`、`BINARY_ADD`...），调用对应的 `binop`、`call_function`、`LOAD_ATTR` 等 visit 方法；
- **逐指令建立 VariableTracker**：每个 Python 对象都被包装成 `VT`，记录其来源（常量、参数、属性、call 结果）；
- 真正调用 PyTorch op 时，调用 `tx.output.create_node` 创建一个 `fx.Node`，并把 `VT` 链到 `proxy` 上。

关键的几个 VT（[`torch/_dynamo/variables/`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_dynamo/variables/)）：

| VariableTracker | 含义 |
|---|---|
| `TensorVariable` | FX 图里的 `placeholder` / `call_function` 节点 |
| `BuiltinVariable` | `torch.sin`、`torch.matmul` 等 |
| `NNModuleVariable` | `nn.Linear` 等模块（按 `__call__` trace） |
| `ListVariable` / `DictVariable` | Python list/dict（结构可变，break graph 的高发区） |
| `UserDefinedObjectVariable` | 自定义类，一般 break graph |

### 2.4 `OutputGraph`：拼装 FX 图 + 收集 guards

`torch/_dynamo/output_graph.py:404` 的 `OutputGraph` 持有：

- `fx.GraphTracer`：FX 的 `Tracer` 子类，负责生成节点；
- `guards`：当前图对应的 guard 集合（详见 §3）；
- `graphargs`：图的输入参数；
- `bytecode`/`instructions`：用于支持 graph break 后的恢复执行。

每个算子的 trace 都会走一遍 `VariableTracker.build_graph_node`，最后调用 `OutputGraph.compile_subgraph`：

```python
# 简化版伪代码
def compile_subgraph(self, instructions, tx):
    g = self.tracer.graph
    # 调用用户注册的 backend（默认是 inductor）生成可调用对象
    compiled_fn = self.backend(g, self.example_inputs)
    # 构造 CheckFunctionManager：每次调用前先跑 guards
    return compiled_fn, check_fn_manager
```

最终返回给 frame hook 的是一个新 code object，里面直接是：

```python
def compiled_frame(args):
    if not check_fn_manager.check(args):
        # guards 失败 → 走 fallback
        return original_frame(args)
    return compiled_fn(*args)
```

### 2.5 Graph break：为什么会出现

Graph break 出现在 `InstructionTranslatorBase.break_graph_if_unsupported`（[`symbolic_convert.py:889`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py)），典型触发条件：

- **数据相关控制流**：例如 `if x.sum() < 0:` —— `x.sum() < 0` 的结果只有运行时知道，Dynamo 无法在 trace 期决定走哪个分支（`generic_jump`，[`symbolic_convert.py:626`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py)）；
- **不支持的 Python 内建**：例如 `print`、`input`、`open`；
- **第三方 C 扩展调用**：没注册 Dynamo 自定义算子的；
- **attribute mutation on user-defined object**：`obj.x = ...`，Dynamo 不知道类型变化；
- **`raise` / `try-except`**：异常分支有时不可 trace。

break 后：

1. 当前已 trace 部分被切成一个 `subgraph`（FX 图）；
2. 字节码被替换为 `compiled_frame` 调用；
3. **未支持的指令由 resume execution**（[`resume_execution.py`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_dynamo/resume_execution.py)）恢复：用新生成的字节码把断点处的 Python 栈恢复，然后回到 eager 模式跑剩余字节码。

下面这个 `bar` 函数：

```python
def bar(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b
```

会被切成 3 个子图：
- 子图 A：`x = a / (torch.abs(a) + 1)`；
- 子图 B：`b = b * -1`（仅 `if` 为真时跑）；
- 子图 C：`return x * b`。

绕开 break 的常见做法（来自你已有的 README）：

```python
# 用 torch.where 替换 if
b = torch.where(b.sum() < 0, b * -1, b)
return x * b
```

这样控制流就成了"纯 tensor 计算"，Dynamo 能合并到一个图里。

---

## 3. Guards：spec 化的"重 trace 触发器"

### 3.1 什么是 guard

每个 trace 出来的 FX 图都附带一组 **guard 表达式**。再次调用时，Dynamo 会先评估 guards：若都成立，直接执行编译产物；否则**重新 trace**（recompile）。

Guards 不是简单布尔表达式，而是结构化的 `Guard` 对象 + 一个 Python 表达式（`code_framelocals_names_reversed_cached`），后者由 `GuardBuilder` 生成。

### 3.2 `GuardBuilder` 的常见 guard 类型

源码在 [`torch/_dynamo/guards.py:961`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_dynamo/guards.py)。常见类型：

| Guard | 触发条件 | 例子 |
|---|---|---|
| `TENSOR_MATCH` | `tensor.shape`、`dtype`、`device`、`stride` | `torch._dynamo.is_matching_tensors(...)` |
| `ID_MATCH` | Python 对象 id 不变 | `id(x) == 12345` |
| `EQUALS_MATCH` | 对象相等 | `x == 5` |
| `TYPE_MATCH` | `type(x) is T` | `type(x) is int` |
| `DICT_KEYS_MATCH` / `DICT_CONTAINS` | dict 结构 | `set(d.keys()) == {...}` |
| `NN_MODULE` | `nn.Module` 类名匹配 | `type(self.linear) is nn.Linear` |
| `DYNAMIC_DIM` | 是否被 `mark_dynamic` | `mark_dynamic(t, 0)` |

### 3.3 `CheckFunctionManager`：guard 评估器

[`guards.py:3070`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_dynamo/guards.py) 的 `GuardsState` + `CheckFunctionManager` 把所有 guard 编译成一个 **Python 函数**（`torch.compile` 生成的 `___check_tensors` 等），每次调用前执行它：

```python
# 伪代码（实际由 GuardBuilder 生成）
def check(x):
    return (x.shape == (3, 3)
            and x.dtype == torch.float32
            and x.device.type == 'cpu'
            and type(self.linear) is torch.nn.Linear)
```

评估为 False 时，会记录"哪些 guard 失败"，触发 recompile。

### 3.4 Recompile 缓存与上限

每个 `(code_object, guards)` 是一个 cache entry。`torch.compile` 有一个默认上限：

```python
# torch/dynamo/config.py
cache_size_limit = 8        # 默认 8 个不同的 guard 组合就触发更多 recompile
accumulated_cache_size_limit = 256
```

超过上限会跑 fallback（不再 compile）或抛 `RecompileLimitExceeded`。常见原因：

- **dynamic shapes 没有声明**：每次 batch size 都不同 → 每个 size 都触发 recompile；
- **nn.Module 类型变化**：把同一个 Python 变量复用于不同类的子模块；
- **dict/list 结构变化**：dict 的 key 集合变了。

修法：
- `torch._dynamo.mark_dynamic(x, 0)` 声明 dynamic dim；
- `torch._dynamo.config.cache_size_limit = 64` 调大上限；
- `torch._dynamo.reset()` 清空缓存（`eval_frame.py:119`）。

---

## 4. AOTAutograd：把 FX 图变成 (前向, 反向) 两张图

### 4.1 为什么需要 AOTAutograd

Dynamo 拿到的是**前向图**。对训练任务，反向梯度也需要被 Inductor 编译。AOTAutograd 做的就是：

1. 把 Dynamo 给的 forward FX 图 **再 trace 一遍反向**（用 `torch.func.grad` / functorch）；
2. 得到一个 **joint forward + backward 图**；
3. 用 partitioner（[`partitioners.py`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_functorch/partitioners.py)）切回两张分离的图，分别交给 `fw_compiler` 和 `bw_compiler`；
4. 中间把不直接被 autograd 支持的 op **降级**（decomposition）到一组核心算子。

源码入口：`torch/_functorch/aot_autograd.py:711` 的 `aot_function`：

```python
def aot_function(fn, fw_compiler, bw_compiler=None, partition_fn=default_partition,
                 decompositions=None, num_params_buffers=0, ...):
    """
    Traces the forward and backward graph of fn using torch dispatch mechanism,
    and then compiles the generated forward and backward graphs through
    fw_compiler and bw_compiler.
    """
```

调用流程（[`aot_autograd.py:711-833`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_functorch/aot_autograd.py)）：

```python
flat_fn, out_spec = create_tree_flattened_fn(fn, args, kwargs)
(fake_mode, shape_env) = construct_fake_mode(flat_args, aot_config)
fake_flat_args = process_inputs(flat_args, aot_config, fake_mode, shape_env)

with ExitStack() as stack:
    aot_state = create_aot_state(...)           # 构造 FakeTensor 模式
    aot_graph_capture = aot_stage1_graph_capture(aot_state, flat_fn)  # trace joint graph
    compiled_fn, _ = aot_stage2_compile(aot_state, aot_graph_capture) # partition + 编译
```

### 4.2 Decomposition：算子降级

训练里常用 `aten._softmax`、`aten.native_dropout`、`aten._fused_dropout` 等 fused op，但 Inductor 不一定认识。AOTAutograd 用 **decomposition table** 把它们降级成更基础、确定性的算子：

- `aten._softmax` → `aten.exp + aten.logsumexp + aten.div`；
- `aten._fused_dropout` → `aten.mul(mask)`；
- `aten.addmm` → `aten.mm + aten.add`（如果目标 backend 没有融合 GEMM）。

这套机制使得 Inductor **只需要实现一组"核心算子"**。

### 4.3 Partitioning：joint graph → forward + backward

[`partitioners.py`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_functorch/partitioners.py) 实现 `default_partition`：

- 在 joint 图上做反向扫描，识别哪些前向 activation 必须保留（被 backward 用到的）；
- 其余 activation 通过 **recomputation** 策略重建（前向里加几个算子，比存 activation 省内存）；
- 切出 forward 和 backward 两个 FX Graph；
- 给 Inductor 时使用 `fw_compiler` 编 forward、`bw_compiler` 编 backward。

高级模式：

- **Activation checkpointing**（`torch.utils.checkpoint`）：把 partitioner 换成 `min_cut_partitioning` 自动选 save/recompute 点；
- **Inference mode**：跳过 backward，仅编 forward。

### 4.4 `aot_module` 与 `aot_module_simplified`

`aot_module`（[`aot_autograd.py:837`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_functorch/aot_autograd.py)）处理 `nn.Module`，把 parameters/buffers 抽出来做静态输入；`aot_module_simplified`（[`aot_autograd.py:1016`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_functorch/aot_autograd.py)）是 `torch.compile` 默认走的路径，把上述所有步骤合一：

```python
def aot_module_simplified(mod, args, fw_compiler, bw_compiler=None, ...):
    """Trace and compile a module with AOTAutograd + a backend."""
```

---

## 5. Inductor：FX 图 → Triton + Python wrapper

### 5.1 入口：`compile_fx`

AOTAutograd 把 forward / backward FX Graph 喂给 Inductor 的 `compile_fx_inner`（[`torch/_inductor/compile_fx.py:743`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_inductor/compile_fx.py)）：

```python
def compile_fx_inner(gm: GraphModule, example_inputs: Sequence[InputType], **kwargs) -> OutputCode:
    # 1. 清掉上次的 Triton 编译缓存
    torch._inductor.async_compile.CompiledTritonKernels.cache_clear()
    # 2. 进入内部实现
    return wrap_compiler_debug(_compile_fx_inner, compiler_name="inductor")(...)
```

`_compile_fx_inner`（[`compile_fx.py:790`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_inductor/compile_fx.py)）的主要步骤：

1. `fx_codegen_and_compile`：把 FX Graph 转成调度计划（`scheduler.py`）和代码模板（`codegen/triton.py`）；
2. 在 fake tensor 上做**几轮 passes**：
   - pre-grad passes：算子融合（`aten.add + aten.relu` → 单个 triton kernel）；
   - post-grad passes：cudagraph 化、内存规划（`memory_planning.py`）；
3. 触发 **autotune**（`autoheuristic/`）：选最快的 BLOCK_SIZE / num_warps；
4. 用 `triton.jit` 编译 Triton 内核到 CUDA；
5. 拼接 Python wrapper（含 cudagraph 捕获）；
6. 返回一个 `CompiledFx` callable。

### 5.2 调度：`torch._inductor.scheduler`

Inductor 的核心 IR 是 **node list**（带依赖关系）。`scheduler` 阶段做：

- **算子融合**（pointwise、reduction、matmul、reduction+pointwise）；
- **Buffer reuse**：分配中间 tensor 时尽量复用其他已死 tensor 的内存；
- **依赖图分析**：把节点分组成 **fusions**（每个 fusion 是一个 Triton kernel）；
- **跨 GPU/CPU 的搬运**（如果是异构场景）。

Fusion 策略大致是：
- 连续若干 elementwise → 融合为一个 triton kernel；
- `matmul + epilogue` (bias add + relu + ...) → 一个 Triton GEMM template；
- 长 reduction + 后续 elementwise → 切为两个 kernel（reduce kernel + epilogue kernel），因为 reduction 需要 cross-block 同步。

### 5.3 Codegen：`torch._inductor.codegen`

每个 backend 一份 codegen：

| 后端 | 文件 | 输出 |
|---|---|---|
| Triton (NVIDIA / AMD) | `codegen/triton.py` | Triton Python 源码，运行时调 `triton.jit` |
| CUDA C++ (CUTLASS) | `codegen/cuda/` | nvcc 编译的 .so |
| CPP wrapper | `codegen/cpp_wrapper_gpu.py` | C++ 调用 Triton/CUDA 的 wrapper |
| Halide | `codegen/halide.py` | Halide schedule（实验） |
| CuteDSL | `codegen/cutedsl/` | NVIDIA Cute DSL（实验） |

最终 wrapper 输出大概长这样（`@triton.jit` kernel + Python glue）：

```python
# 简化示意，由 codegen 拼接出来
@triton.jit
def triton_red_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, out, mask=mask)

def call(x):
    n = x.numel()
    out = torch.empty(BLOCK_N, dtype=x.dtype, device=x.device)
    triton_red_kernel[(triton.cdiv(n, BLOCK_SIZE),)](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out.sum()
```

### 5.4 自定义 Triton 算子：`@torch.library.custom_op`

Inductor 默认会调 aten 的 Triton 实现，但用户可以插入自己的 kernel。两种方式：

**方式一：`torch._inductor.ir` + `custom_op`**（高级，写 IR）：

```python
from torch._inductor.ir import Pointwise
from torch._inductor.lowering import register_lowering

@torch.library.custom_op("mylib::my_softmax", mutates_args=())
def my_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)

@my_softmax.register_fake
def _(x):
    return torch.empty_like(x)

@register_lowering(torch.ops.mylib.my_softmax.default, type_promotion_kind=None)
def _my_softmax(x):
    # 返回一个 Inductor IR 节点，可以是 TritonTemplate 或 Pointwise
    return Pointwise.create(...)
```

**方式二：`triton.jit` + `aot_autograd`（更常见）**：写一个 Triton kernel，包装成 Python 函数，Inductor 会通过 `mode="reduce-overhead"` 或 `cudagraphs` 把它纳入。

**方式三（最直接）：让 `torch.compile` 看到算子**。你的 README 里那张 `image.png`（`05-framework/pytorch/compile/image.png`）就是这个流程：自定义 triton kernel → 通过 `@torch.library.custom_op` + `register_fake` 注册 → `torch.compile` 自动 fallback 到你给的实现。

---

## 6. Runtime：guard 检查 + cudagraphs

### 6.1 `OptimizedModule` 的生命周期

每次 `compiled_fn(x)` 走到 frame hook（`eval_frame.py:590` 的 `_TorchDynamoContext`）：

1. 看 cache 里是否有 `(code, guards)` 匹配的编译产物；
2. 有 → 执行其前置 guard 检查（`CheckFunctionManager.check`）：
   - 通过 → 直接调编译产物；
   - 失败 → 走 fallback（重 trace 或原始 eager）；
3. 没有 → 触发第一次 trace + compile，结果塞进 cache。

### 6.2 cudagraphs：消除 Python 与 CUDA launch 的开销

Inductor 在 codegen 后还可以对生成的 wrapper 套一层 **CUDA Graph** 捕获，把整张图的所有 kernel 一次性 launch。源码在 [`compile_fx.py:1733 cudagraphify`](../../../../.conda/envs/pytorch/lib/python3.11/site-packages/torch/_inductor/compile_fx.py) 和 `cudagraph_trees.py`：

- 第一次调用：捕获所有 kernel launch 到 cudagraph；
- 后续调用：replay cudagraph，节省 Python overhead；
- 形状/输入 tensor 改变 → 重新捕获。

启用：

```python
torch._inductor.config.triton.cudagraphs = True
# 或者
@torch.compile(mode="reduce-overhead")
def f(x): ...
```

`mode="max-autotune"` 还会自动选 kernel 参数 / 切分。

---

## 7. 端到端调用栈：一张图把所有组件串起来

```
用户调用: opt_foo2(x, y)
   │
   ▼
torch._dynamo.eval_frame.OptimizedModule.__call__(x, y)
   │   ←── CPython frame eval hook 已替换
   ▼
DynamoCache.lookup(code_obj) → 命中？(guards)
   │
   ├─ miss → convert_frame.run(code, ...):
   │     │
   │     ▼ InstructionTranslatorBase
   │     │   对每条 bytecode 指令：
   │     │     - VariableTracker.build_graph_node → OutputGraph.create_node
   │     │     - 不支持? → break_graph → ResumeFunction
   │     │
   │     ▼ OutputGraph.compile_subgraph
   │     │   调 backend (default = inductor) 生成 compiled_fn
   │     │   生成 CheckFunctionManager (guards)
   │     │
   │     ▼ 生成新 code object：if not guard_check(args): fallback; else compiled_fn(args)
   │
   ▼
执行新 code object
   │
   ├─ guard pass → compiled_fn(x, y)
   │     │
   │     ▼ AOTAutograd (第一次 trace)
   │     │   aot_module_simplified:
   │     │     - construct_fake_mode
   │     │     - aot_stage1_graph_capture: trace joint fwd+bwd
   │     │     - partition (default_partition): fwd graph + bwd graph
   │     │     - decompositions: 把 fused op 降级
   │     │
   │     ▼ Inductor (compile_fx_inner → _compile_fx_inner)
   │     │   - pre_grad passes (fusion, memory planning)
   │     │   - scheduler (fusion groups)
   │     │   - codegen/triton.py → Triton kernel 源码
   │     │   - triton.jit → CUDA .cubin
   │     │   - codegen Python wrapper
   │     │   - (可选) cudagraph 捕获
   │     │
   │     ▼ OutputCode: callable wrapper
   │
   └─ guard fail → run_only stance → 直接调原 eager
```

---

## 8. 各组件职责速查表

| 组件 | 输入 | 输出 | 关键产物 |
|---|---|---|---|
| `OptimizedModule.__call__` | 用户调用 | 调用结果 | 触发 trace / 重用 cache |
| `InstructionTranslator` | Python 字节码 | 走完所有指令 / break | 中间 VT 状态 |
| `OutputGraph` | VT + 节点创建请求 | FX Graph + guards | FX GraphModule, CheckFunctionManager |
| `CheckFunctionManager` | 下次调用的 args | True / False | 触发 recompile 或直接复用 |
| `aot_module_simplified` | FX GraphModule + example_inputs | 编译后的 forward+backward | 包装好的 `compiled_fn` |
| `default_partition` | joint fwd+bwd graph | (fwd graph, bwd graph) | 给 Inductor 的两张图 |
| `compile_fx_inner` | FX GraphModule | `OutputCode` | Python wrapper + Triton kernels |
| `scheduler` | FX graph (post-decompose) | schedule plan | fusion groups, memory plan |
| `codegen/triton.py` | schedule | Triton source + wrapper | `@triton.jit` functions |
| `cudagraphify` | wrapper | wrapped with cudagraph | cudagraph replay |

---

## 9. 常用配置与诊断 API

```python
# 关键开关
import torch._dynamo as dynamo
import torch._inductor.config as ic

dynamo.config.cache_size_limit = 64          # 调大 recompile 上限
dynamo.config.accumulated_cache_size_limit = 256
dynamo.config.suppress_errors = False        # True 时 break 后静默回退到 eager

ic.triton.cudagraphs = True                  # 启用 cudagraphs
ic.max_autotune = True                       # autotune kernel 参数
ic.coordinate_descent_tuning = True          # 切分/融合参数搜索
ic.epilogue_fusion = True                    # bias+relu 等 epilogue 融合

# 诊断
torch._logging.set_logs(output_code=True,    # 打印编译产物
                        graph_code=True,
                        graph_breaks=True,
                        recompiles=True,
                        guards=True)
TORCH_LOGS="+dynamo,+inductor" python train.py   # CLI

# 控制 dynamic shape
torch._dynamo.mark_dynamic(x, 0)             # 声明 dim 0 是 dynamic
torch._dynamo.mark_static(x, 0)              # 声明 dim 0 是 static

# 手动干预图捕获
torch._dynamo.allow_in_graph(my_function)    # 不 trace 这个函数
torch._dynamo.disallow_in_graph(my_function) # 强制 trace
```

---

## 10. 常见坑 & 修复

| 问题 | 原因 | 修法 |
|---|---|---|
| 频繁 recompile | 输入 shape 变化 / 控制流没标 dynamic | `mark_dynamic`、调大 cache_size_limit |
| Graph break 太多 | 数据相关 `if`、不支持的 C 扩展、attribute 突变 | 用 `torch.where`、注册 `custom_op`、避免修改 user object |
| 性能反而变慢 | 图太小，launch overhead 占主导 | 开 `cudagraphs`；用 `mode="reduce-overhead"` |
| DDP 报错 | Dynamo 在前向 trace 里捕获了 collectives | `@torch._dynamo.allow_in_graph(torch.distributed.all_reduce)` |
| 内存爆 | AOTAutograd 默认存所有 activation | `torch.utils.checkpoint` + `min_cut_partition` |
| 第一次巨慢 | AOTAutograd + Inductor 全程编译 | 预热（warmup）几个 batch 再正式训练 |
| `torch.compile` 不动 | wrap 在 `torch.no_grad()` 之外但函数是 inference | `torch.inference_mode()` 或 `@torch.compile` 直接装饰 nn.Module |

---

## 11. 与 PyTorch 其他编译路径的关系

```
torch.compile          ← 默认入口（Dynamo + AOTAutograd + Inductor）
   │
   ├─ 后端可替换：
   │     backend="inductor"        # 默认
   │     backend="aot_eager"       # 只跑 AOTAutograd，不下放到 Triton
   │     backend="eager"           # 跳过编译，等价 disable
   │     backend=cudagraphs_fn     # 只做 cudagraph 捕获
   │     backend=my_custom_backend # 自己写 fx → callable 的函数
   │
   ├─ 与其他子系统的关系：
   │     - torch.export:           不走 Dynamo，走 TorchDynamo 的 export 模式（torch.export）
   │     - torch.fx:               Dynamo 输出就是 FX GraphModule
   │     - torch._dynamo:          主入口同名包；其他后端（nvFuser、TensorRT）也是 Dynamo 的 backend
   │     - torch.compile + DTensor: Dynamo 不会破坏 DTensor 分片，但 Inductor 会调 DTensor 的 collective lowering
```

---

## 12. 推荐阅读源码的顺序

1. `torch/_dynamo/eval_frame.py` 看 `OptimizedModule.__call__`；
2. `torch/_dynamo/convert_frame.py` 看 `ConvertFrameAssert.process` 怎么进 symbolic_convert；
3. `torch/_dynamo/symbolic_convert.py` 看 `InstructionTranslator` 的几个常见 visit；
4. `torch/_dynamo/output_graph.py` 看 FX Graph + guards 怎么产生；
5. `torch/_dynamo/guards.py` 看 `GuardBuilder` 生成 guard 表达式的过程；
6. `torch/_functorch/aot_autograd.py` 看 `aot_module_simplified`；
7. `torch/_functorch/partitioners.py` 看 `default_partition`；
8. `torch/_inductor/compile_fx.py` 看 `_compile_fx_inner`；
9. `torch/_inductor/scheduler.py` 看 fusion group 是怎么划分的；
10. `torch/_inductor/codegen/triton.py` 看 Triton codegen 的模板和拼接逻辑。

配合 `TORCH_LOGS="graph_code,graph_breaks,recompiles"` 实际跑一遍 `demo.py`，会看到完整流水线。
