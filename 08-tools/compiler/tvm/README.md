# TVM

TVM 是一个用于CPU，GPU和机器学习平加速器的开源机器学习编译器框架，可以针对不同的后端进行优化和计算。

TVM的步骤如下所示

![alt text](image.png)

- 从Tensorflow等框架中导入模型，
- 翻译为TVM的高级语言Relay
- 降级为张量表达式(TE)，使用较为低级的表示，
- 使用auto-tuning等调优模块来搜索最佳的调度方案
- 为模型编译选择最佳配置
- 降级为张量中间表示(TIR)，通过底层优化pass去进行优化】
- 编译成机器码，并生成可执行文件

## 1. example

先使用relax.frontend来构建一个简单的模型

```python
import tvm
from tvm import relax
from tvm.relax.frontend import nn


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
```

这是一个两层的MLP模型，继承自`relax`的`nn.module`

使用`export_tvm`方法将模型导出为TVM的`IRModule`

```python
mod, param_spec = MLPModel().export_tvm(
    spec={"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
)
mod.show()
```

输出的IR 如下所示

```python
@I.ir_module
class Module:
    @R.function
    def forward(x: R.Tensor((1, 784), dtype="float32"), fc1_weight: R.Tensor((256, 784), dtype="float32"), fc1_bias: R.Tensor((256,), dtype="float32"), fc2_weight: R.Tensor((10, 256), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            permute_dims: R.Tensor((784, 256), dtype="float32") = R.permute_dims(fc1_weight, axes=None)
            matmul: R.Tensor((1, 256), dtype="float32") = R.matmul(x, permute_dims, out_dtype="void")
            add: R.Tensor((1, 256), dtype="float32") = R.add(matmul, fc1_bias)
            relu: R.Tensor((1, 256), dtype="float32") = R.nn.relu(add)
            permute_dims1: R.Tensor((256, 10), dtype="float32") = R.permute_dims(fc2_weight, axes=None)
            matmul1: R.Tensor((1, 10), dtype="float32") = R.matmul(relu, permute_dims1, out_dtype="void")
            gv: R.Tensor((1, 10), dtype="float32") = matmul1
            R.output(gv)
        return gv
```

对模型进行优化，优化分为两个层级
- 模型层级: 进行算子融合，布局转换等
- 张量层级：进行底层的优化，例如更换数学库，优化代码生成

模型运行

```python
import numpy as np

target = tvm.target.Target("llvm")
ex = tvm.compile(mod, target)
device = tvm.cpu()
vm = relax.VirtualMachine(ex, device)
data = np.random.rand(1, 784).astype("float32")
tvm_data = tvm.nd.array(data, device=device)
params = [np.random.rand(*param.shape).astype("float32") for _, param in param_spec]
params = [tvm.nd.array(param, device=device) for param in params]
print(vm["forward"](tvm_data, *params).numpy())
```

## 2. Tensor IR

### 2.1 基础语法

```python3
@T.prim_func
def vector_add(A: T.Buffer[128, "float32"],
               B: T.Buffer[128, "float32"],
               C: T.Buffer[128, "float32"]):
    # 并行执行元素级加法
    for i in T.parallel(128):
        with T.block("C"):
            vi = T.axis.spatial(128, i)
            C[vi] = A[vi] + B[vi]
```

`@T.prim_func` 是对该函数的声明

`T.block` 其中的Block是TensorIR中的基本计算单位

对于for循环，可以使用 `T.grid` 这种语法糖进行处理

```python
for i, j, k in T.grid(128, 128, 128):
    with T.block("Y"):
```

对于轴块绑定，使用 `T.axis.remap`声明轴块，S表示范围，R表示归约。

```python
vi, vj, vk = T.axis.remap("SSR", [i, j, k])
```

### 2.2 并行化/向量化/循环展开

如下所示，相对i循环进行切分，将其划分为i0和i1两层，再对i0层进行并行，对i1层进行循环展开，对j层进行并行化。

```python3
@tvm.script.ir_module
class MyAdd:
    @T.prim_func
    def add(A: T.Buffer((4, 4), "int64"),
            B: T.Buffer((4, 4), "int64"),
            C: T.Buffer((4, 4), "int64")):
        T.func_attr({"global_symbol": "add", "tir.noalias": True})
        for i, j in T.grid(4, 4):
            with T.block("C"):
                vi = T.axis.spatial(4, i)
                vj = T.axis.spatial(4, j)
                C[vi, vj] = A[vi, vj] + B[vi, vj]

sch = tvm.tir.Schedule(MyAdd)
block = sch.get_block("C", func_name="add")
i, j = sch.get_loops(block)
i0, i1 = sch.split(i, factors=[2, 2])
sch.parallel(i0)
sch.unroll(i1)
sch.vectorize(j)
print(IPython.display.Code(sch.mod.script(), language="python"))
```




