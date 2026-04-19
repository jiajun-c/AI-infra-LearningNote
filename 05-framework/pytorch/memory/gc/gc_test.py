import torch
import torch.nn as nn
import gc
import sys
import time


def print_memory_info(prefix=""):
    """打印当前内存使用情况"""
    gc.collect()
    allocated = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    reserved = torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0
    print(f"[{prefix}] GPU: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")
    return allocated, reserved


def get_refcount(obj):
    """获取对象的引用计数"""
    return sys.getrefcount(obj)


# ============================================================
# 情况 1: 普通对象 - del 后自动释放
# ============================================================
def test_normal_object():
    print("\n" + "=" * 60)
    print("情况 1: 普通对象 - del 后自动释放")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过 GPU 测试")
        return

    print_memory_info("初始")

    # 创建大张量
    x = torch.randn(10000, 1000).cuda()
    print(f"创建张量后，引用计数：{sys.getrefcount(x)}")
    print_memory_info("创建张量后")

    # del 后应该自动释放
    del x
    print_memory_info("del 后 (应该自动释放)")

    # 再调用 gc.collect() 和 empty_cache
    gc.collect()
    torch.cuda.empty_cache()
    print_memory_info("gc.collect() + empty_cache 后")


# ============================================================
# 情况 2: 循环引用 - del 后不会自动释放
# ============================================================
class Node:
    """带有循环引用的节点类"""
    def __init__(self, name):
        self.name = name
        self.parent = None
        self.children = []
        # 创建一个大张量来模拟 GPU 内存占用
        if torch.cuda.is_available():
            self.data = torch.randn(5000, 500).cuda()
        else:
            self.data = None

    def __del__(self):
        print(f"  -> Node '{self.name}' 被销毁")


def test_circular_reference():
    print("\n" + "=" * 60)
    print("情况 2: 循环引用 - del 后不会自动释放")
    print("=" * 60)

    if torch.cuda.is_available():
        print_memory_info("初始")

    # 创建循环引用
    print("创建循环引用对象...")
    a = Node("A")
    b = Node("B")
    a.children.append(b)
    b.parent = a  # 形成循环引用：a -> b -> a

    if torch.cuda.is_available():
        print_memory_info("创建循环引用后")

    # del 外部引用
    print("执行 del a, b ...")
    del a, b

    if torch.cuda.is_available():
        print_memory_info("del 后 (循环引用未释放!)")

    # 必须调用 gc.collect()
    print("执行 gc.collect()...")
    gc.collect()

    if torch.cuda.is_available():
        print_memory_info("gc.collect() 后 (应该释放了)")


# ============================================================
# 情况 3: 闭包/回调持有引用
# ============================================================
def test_closure_reference():
    print("\n" + "=" * 60)
    print("情况 3: 闭包/回调持有引用")
    print("=" * 60)

    if torch.cuda.is_available():
        print_memory_info("初始")

    # 创建带闭包的回调列表
    callbacks = []
    large_objects = []

    print("创建带闭包的回调...")
    for i in range(5):
        # 每个闭包都持有 large_obj 的引用
        large_obj = torch.randn(3000, 300).cuda() if torch.cuda.is_available() else [0] * 10000
        large_objects.append(large_obj)

        def callback(x=i, obj=large_obj):
            return f"Callback {x} with obj id: {id(obj)}"

        callbacks.append(callback)

    if torch.cuda.is_available():
        print_memory_info("创建闭包后")

    # 尝试清理
    print("执行 del large_objects...")
    del large_objects

    if torch.cuda.is_available():
        print_memory_info("del large_objects 后 (闭包仍持有引用!)")

    # 清空回调列表
    print("执行 callbacks.clear()...")
    callbacks.clear()

    if torch.cuda.is_available():
        print_memory_info("callbacks.clear() 后")

    gc.collect()
    torch.cuda.empty_cache()
    print_memory_info("gc.collect() + empty_cache 后")


# ============================================================
# 情况 4: 类属性循环引用
# ============================================================
class ModelWrapper:
    """包装模型，形成循环引用"""
    def __init__(self, model):
        self.model = model
        model.wrapper = self  # 反向引用！
        self.buffer = torch.randn(2000, 200).cuda() if torch.cuda.is_available() else None

    def __del__(self):
        print(f"  -> ModelWrapper 被销毁")


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(100, 100)
        self.wrapper = None  # 会被 wrapper 反向引用

    def __del__(self):
        print(f"  -> SimpleModel 被销毁")


def test_model_circular_reference():
    print("\n" + "=" * 60)
    print("情况 4: 模型包装器循环引用")
    print("=" * 60)

    if torch.cuda.is_available():
        print_memory_info("初始")

    print("创建模型和包装器（形成循环引用）...")
    model = SimpleModel().cuda() if torch.cuda.is_available() else SimpleModel()
    wrapper = ModelWrapper(model)

    if torch.cuda.is_available():
        print_memory_info("创建模型后")

    print("执行 del model, wrapper ...")
    del model, wrapper

    if torch.cuda.is_available():
        print_memory_info("del 后 (循环引用未释放!)")

    print("执行 gc.collect()...")
    gc.collect()

    if torch.cuda.is_available():
        print_memory_info("gc.collect() 后")


# ============================================================
# 情况 5: 长运行脚本中的内存积累
# ============================================================
def test_long_running_script():
    print("\n" + "=" * 60)
    print("情况 5: 长运行脚本模拟（内存积累）")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过")
        return

    print_memory_info("初始")

    # 模拟多个批次的处理
    for i in range(5):
        # 模拟一些临时对象
        data = torch.randn(2000, 500).cuda()
        result = data @ data.t()
        del data, result

        # 不调用 gc.collect() 的情况
        print(f"批次 {i+1} 完成 (未调用 gc.collect())")

    print_memory_info("5 个批次后 (未调用 gc.collect())")

    # 定期清理
    gc.collect()
    torch.cuda.empty_cache()
    print_memory_info("调用 gc.collect() 后")


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "#" * 60)
    print("#  Python GC 测试 - 不同引用场景的内存释放行为")
    print("#" * 60)

    if not torch.cuda.is_available():
        print("\n警告：CUDA 不可用，将跳过 GPU 内存测试")

    test_normal_object()
    test_circular_reference()
    test_closure_reference()
    test_model_circular_reference()
    test_long_running_script()

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
