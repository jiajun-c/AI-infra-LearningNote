# __new__ and __init__

## __new__

负责进行类的创建和内存分配，在类初始化的第一步进行调用，其本质是一个静态方法，其接收的是类本身，根据这个类去生成实例，其必须要有返回值


## __init__

负责进行类的初始化，在类完成分配后执行，其是一个实例方法，接收的是实例本身 (self)，此时对象已经被创建出来了。

如下所示

```python
class LifeCycleDemo:
    def __new__(cls, *args, **kwargs):
        print(f"[1] __new__ 被调用。接收到的类是: {cls.__name__}")
        # 必须手动调用 super().__new__ 来真正在内存中创建对象
        instance = super().__new__(cls) 
        print(f"    -> 对象已在内存中创建，地址: {hex(id(instance))}")
        return instance # 把毛坯房交出去

    def __init__(self, name):
        print(f"[2] __init__ 被调用。接收到的 self 地址: {hex(id(self))}")
        # 开始装修毛坯房
        self.name = name
        print(f"    -> 属性初始化完成。")

# 触发实例化
obj = LifeCycleDemo("MolFM_Model")
```

## `__new__` 的进阶用法

详见 [new_feature.py](new_feature.py)，包含以下 5 个实战场景：

### 1. 单例模式（Singleton）

全局只创建一个实例，适用于配置管理器、日志器、GPU 设备句柄等全局唯一资源。

### 2. 不可变类型子类化

继承 `int`/`str`/`tuple` 等不可变类型时，值在 `__init__` 阶段已经固定，只能在 `__new__` 中修改。例如创建只允许偶数的 `EvenInt`、自动转大写的 `UpperStr`。

### 3. 对象缓存池（享元模式）

相同参数复用已有实例，避免重复创建。类似于 Python 内部对小整数 (-5~256) 的缓存机制，适用于 CUDA stream 管理、Tensor dtype 对象复用等场景。

### 4. 工厂模式

`__new__` 根据参数返回不同子类的实例，例如根据 `device` 参数自动创建 `CPUTensor` 或 `CUDATensor`。

### 5. 实例数量限制

限制类最多创建 N 个实例，适用于连接池、worker 数量限制等场景。

## 核心对比

| | `__new__` | `__init__` |
|---|---|---|
| 职责 | 创建实例（分配内存） | 初始化实例（设置属性） |
| 类型 | 静态方法 | 实例方法 |
| 第一个参数 | `cls`（类本身） | `self`（实例本身） |
| 返回值 | 必须返回实例 | 必须返回 None |
| 调用时机 | 实例创建之前 | 实例创建之后 |

- 需要控制 **"是否创建、创建什么"** → 用 `__new__`
- 需要控制 **"怎么初始化"** → 用 `__init__`