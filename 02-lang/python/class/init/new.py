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