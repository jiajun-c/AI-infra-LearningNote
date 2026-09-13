# Python类

## dataclass

Python的dataclass其原理类似C++的聚合初始化 + 自动生成的 `operator==`，帮我们生成一批协议方法。

**注意：dataclass 不生成 getter/setter。** C++ 里写 getter/setter 是为了访问控制和保持接口稳定（字段私有，改内部表示不影响外部）；Python 属性访问本来就是动态的，`obj.x` 已经是一次 `__getattribute__` 调用，加不加 property 调用方的写法不变。真正需要计算或校验时再补 `property`，且调用方一行都不用改。

### 生成的方法

|方法|条件|
|---|---|
|`__init__`|`init=True`（默认）|
|`__repr__`|`repr=True`（默认）|
|`__eq__`|`eq=True`（默认），且类里未自己定义 `__eq__`|
|`__lt__` `__le__` `__gt__` `__ge__`|`order=True`，**默认 False**|
|`__hash__`|见下|
|`__match_args__`|Python 3.10+，供 `match` 用|
|`__slots__`|`slots=True`|

`__hash__` 分三种情况，容易踩：

- `eq=True, frozen=True` → 生成 `__hash__`（按字段元组哈希）
- `eq=True, frozen=False` → **`__hash__ = None`，类不可哈希**，不能进 `set` / 当 dict 键
- `eq=False` → 不动，继承 `object.__hash__`

### 可变默认值

`field(default_factory=list)` 不能写成 `= []`：可变默认值在**类定义时求值一次**，所有实例共享同一个 list，`a.items.append(1)` 会污染 `b.items`。dataclass 对 `list/dict/set` 默认值直接抛 `ValueError` 拦住你。

普通函数参数 `def f(x=[])` 是同一个坑。

## 属性查找顺序

`obj.x` 实际执行 `type(obj).__getattribute__(obj, 'x')`，CPython 的顺序：

```text
1. 在 type(obj).__mro__ 里找 'x'              → 记为 descr
2. descr 存在且其类型定义了 __get__：
     若它同时是「数据描述符」（有 __set__ / __delete__）：
         → 调 descr.__get__ 返回       ★ 压过实例字典
3. 在 obj.__dict__ 里找 'x' → 找到就返回
4. descr 存在：
     有 __get__ → 调 descr.__get__ 返回   ← 非数据描述符在这
     没有 __get__ → 直接返回 descr        ← 普通类属性
5. 调 __getattr__('x')（若定义）
6. 抛 AttributeError
```

**数据描述符 vs 非数据描述符**，判别标准是看有没有定义 `__set__`：

|类型|例子|优先级|
|---|---|---|
|数据描述符|`property`（有 `__get__`+`__set__`）|**高于**实例字典|
|非数据描述符|普通函数、`classmethod`|低于实例字典|
|非描述符|普通类属性 `x = 1`|垫底|

两个能直接用的推论：

- `property` 能挡住实例字典 → `obj.x = 5` 走 setter，所以 property 能做校验
- 函数是非数据描述符 → `obj.method = 5` 会**盖掉**方法

## PyTorch：nn.Module 为什么要重写 __setattr__

`self.weight = nn.Parameter(...)` 如果走默认的 `object.__setattr__`，weight 只会躺进 `self.__dict__`，于是：

- `state_dict()` 看不到它 → 存不了、加载不了
- `.cuda()` / `.to(dtype)` 不会搬它
- optimizer 收不到它 → **参数永远不会更新**

所以 `nn.Module.__setattr__` 把 Parameter 同时登记进 `self._parameters`，再调 `object.__setattr__` 放进 `__dict__`。

`nn.Module.__getattr__` 是**兜底**（只在正常查找失败时触发），再去 `_parameters` / `_buffers` / `_modules` 里找一遍。
