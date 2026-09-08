# torch compile

## 1. 流程

`torch.compile(fn)` -> Dynamo(字节码->FX图->guards) -> AOTAutograd(joint fwd/bwd -> 分解+切分) -> inductor(FX图->Triton + Python wrapper) -> Runtime(guard检查+cudagraphs)

torch compile有四个核心组件

## 2. 四个核心组件

### 2.1 Dynamo

逐字节码用`InstructionTranslator`做符号化执行，构造FX Graph和guards

### 2.2 Guards

每个 FX 图附带一组结构化 guard（TENSOR_MATCH / ID_MATCH / TYPE_MATCH 等），guard 不过则重新 trace

### 2.3 AOTAutograd

用 functorch 的 FakeTensor trace 出 joint fwd+bwd 图，用 decomposition 表把 fused op 降级，按 partitioner 切回两张图

### 2.4 inductor

- 连续 elementwise 融合为单个 Triton kernel；
- matmul + epilogue 用 Triton GEMM template；
- reduction 与后续 elementwise 切成两个 kernel（reduction 需要 cross-block 同步）；
- 启用 triton.cudagraphs = True 后还会捕获 cudagraph 消除 Python launch overhead。

### 2.5 自定义triton算子输入

@torch.library.custom_op + register_fake (fake tensor用) + @register_lowering(写inductor IR)


## 3 torch compile几种模式

### 3.1 default 

默认模式，做了下面的优化

- kernel fusion：将epilogue阶段融合到规约算子后面
- memroy planning：进行buffer的复用
- Constant folding：trace 期静态求值。

### 3.2 max-autotune-no-cudagraphs

- max_autotune = True — 走多 backend 多候选 benchmark
- coordinate_descent_tuning = True — 围绕最优 config 做精细 ±1 调整


### 3.3 reduce-overhead

开启cudagraphds = True，同时在torch中也有cudagraphtrees的用法，可以当每次变一次shape的时候就重新capture，然后以树的形式保存新的一段路径

