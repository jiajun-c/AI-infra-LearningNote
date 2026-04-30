"""
__new__ 的进阶用法示例

__new__ 真正发挥威力的场景：
1. 单例模式（Singleton）
2. 不可变类型子类化（继承 int/str/tuple 等）
3. 对象缓存池（Flyweight 享元模式）
4. 工厂模式（根据参数返回不同子类实例）
5. 实例数量限制
"""


# ============================================================
# 1. 单例模式 —— 全局只创建一个实例
# ============================================================
# 应用场景: 配置管理器、日志器、GPU 设备句柄等全局唯一资源

class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            print(f"[Singleton] 首次创建实例")
            cls._instance = super().__new__(cls)
        else:
            print(f"[Singleton] 返回已有实例")
        return cls._instance

    def __init__(self, value=None):
        # 注意: __init__ 每次都会被调用，即使返回的是同一个实例
        if value is not None:
            self.value = value


print("=" * 60)
print("1. 单例模式")
print("=" * 60)
a = Singleton("first")
b = Singleton("second")
print(f"a is b: {a is b}")        # True —— 同一个对象
print(f"a.value = {a.value}")      # "second" —— __init__ 被第二次调用覆盖了
print(f"id(a) == id(b): {id(a) == id(b)}")
print()


# ============================================================
# 2. 不可变类型子类化 —— 只能在 __new__ 中修改值
# ============================================================
# int/str/tuple/frozenset 等不可变类型, 在 __init__ 时值已经固定
# 要修改它们的值，必须在 __new__ 阶段截获

class EvenInt(int):
    """只允许偶数的整数类型，奇数自动 +1 变成偶数"""

    def __new__(cls, value):
        value = int(value)
        if value % 2 != 0:
            print(f"[EvenInt] {value} 是奇数，自动调整为 {value + 1}")
            value += 1
        instance = super().__new__(cls, value)
        return instance


class UpperStr(str):
    """创建时自动转为大写的字符串"""

    def __new__(cls, content=""):
        instance = super().__new__(cls, content.upper())
        return instance


print("=" * 60)
print("2. 不可变类型子类化")
print("=" * 60)
e1 = EvenInt(3)
e2 = EvenInt(4)
print(f"EvenInt(3) = {e1}")   # 4
print(f"EvenInt(4) = {e2}")   # 4
print(f"e1 + e2 = {e1 + e2}")  # 8

s = UpperStr("hello cuda")
print(f'UpperStr("hello cuda") = "{s}"')  # "HELLO CUDA"
print(f"类型: {type(s)}")  # <class '__main__.UpperStr'>
print()


# ============================================================
# 3. 对象缓存池（享元模式）—— 相同参数复用已有实例
# ============================================================
# 应用场景: CUDA stream 管理、Tensor dtype 对象复用
# 类似于 Python 内部对小整数 (-5~256) 的缓存机制

class CachedDevice:
    """模拟 GPU 设备对象的缓存，相同 device_id 返回同一实例"""
    _cache = {}

    def __new__(cls, device_id: int):
        if device_id in cls._cache:
            print(f"[CachedDevice] 缓存命中: cuda:{device_id}")
            return cls._cache[device_id]
        print(f"[CachedDevice] 新建设备对象: cuda:{device_id}")
        instance = super().__new__(cls)
        instance.device_id = device_id  # 在 __new__ 中设置，避免 __init__ 重复初始化
        instance.name = f"cuda:{device_id}"
        cls._cache[device_id] = instance
        return instance

    def __init__(self, device_id: int):
        # 因为缓存命中时也会调用 __init__，这里可以留空或做幂等操作
        pass

    def __repr__(self):
        return f"CachedDevice({self.name})"


print("=" * 60)
print("3. 对象缓存池（享元模式）")
print("=" * 60)
dev0_a = CachedDevice(0)
dev0_b = CachedDevice(0)  # 缓存命中
dev1 = CachedDevice(1)    # 新建
print(f"dev0_a is dev0_b: {dev0_a is dev0_b}")  # True
print(f"dev0_a is dev1: {dev0_a is dev1}")        # False
print(f"缓存内容: {CachedDevice._cache}")
print()


# ============================================================
# 4. 工厂模式 —— __new__ 返回不同子类的实例
# ============================================================
# 应用场景: 根据配置自动选择后端 (CPU/CUDA/XPU)

class Tensor:
    """根据 device 参数，自动创建对应设备上的 Tensor 子类"""

    def __new__(cls, data, device="cpu"):
        if cls is not Tensor:
            # 子类直接创建，不再分发
            return super().__new__(cls)
        # 父类根据 device 分发到不同子类
        if device == "cpu":
            return super().__new__(CPUTensor)
        elif device.startswith("cuda"):
            return super().__new__(CUDATensor)
        else:
            raise ValueError(f"不支持的设备: {device}")

    def __init__(self, data, device="cpu"):
        self.data = data
        self.device = device


class CPUTensor(Tensor):
    def compute(self):
        return f"[CPU] 计算 {self.data}"


class CUDATensor(Tensor):
    def compute(self):
        return f"[CUDA] 并行计算 {self.data}"


print("=" * 60)
print("4. 工厂模式")
print("=" * 60)
t1 = Tensor([1, 2, 3], device="cpu")
t2 = Tensor([4, 5, 6], device="cuda:0")
print(f"t1 类型: {type(t1).__name__}, {t1.compute()}")
print(f"t2 类型: {type(t2).__name__}, {t2.compute()}")
print(f"isinstance(t1, Tensor): {isinstance(t1, Tensor)}")  # True
print(f"isinstance(t2, Tensor): {isinstance(t2, Tensor)}")  # True
print()


# ============================================================
# 5. 实例数量限制 —— 限制类最多创建 N 个实例
# ============================================================
# 应用场景: 连接池、worker 数量限制

class LimitedInstances:
    """最多允许创建 max_instances 个实例"""
    _instances = []
    max_instances = 3

    def __new__(cls, name):
        if len(cls._instances) >= cls.max_instances:
            raise RuntimeError(
                f"实例数量已达上限 ({cls.max_instances})，"
                f"无法创建新实例 '{name}'"
            )
        instance = super().__new__(cls)
        cls._instances.append(instance)
        return instance

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Worker({self.name})"


print("=" * 60)
print("5. 实例数量限制")
print("=" * 60)
workers = []
for i in range(3):
    w = LimitedInstances(f"worker-{i}")
    workers.append(w)
    print(f"创建成功: {w}")

try:
    w4 = LimitedInstances("worker-3")  # 超出限制
except RuntimeError as e:
    print(f"创建失败: {e}")
print()


# ============================================================
# 总结
# ============================================================
print("=" * 60)
print("总结: __new__ vs __init__")
print("=" * 60)
print("""
┌─────────────┬───────────────────────┬─────────────────────────┐
│             │      __new__          │       __init__           │
├─────────────┼───────────────────────┼─────────────────────────┤
│ 职责        │ 创建实例 (分配内存)    │ 初始化实例 (设置属性)    │
│ 类型        │ 静态方法              │ 实例方法                 │
│ 第一个参数  │ cls (类本身)          │ self (实例本身)          │
│ 返回值      │ 必须返回实例          │ 必须返回 None            │
│ 调用时机    │ 实例创建之前          │ 实例创建之后             │
│ 典型用途    │ 单例/缓存/工厂/       │ 设置属性值               │
│             │ 不可变类型子类化       │                         │
└─────────────┴───────────────────────┴─────────────────────────┘

核心原则:
  - 当你需要控制"是否创建、创建什么"时 → 用 __new__
  - 当你需要控制"怎么初始化"时 → 用 __init__
""")
