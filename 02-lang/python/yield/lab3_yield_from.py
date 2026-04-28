"""
yield Lab 3：yield from —— 委托与展平
=======================================
目标：理解 yield from 的三个作用：
  1. 展平嵌套迭代器（透明地转发所有值）
  2. 双向通道透传（send/throw/close 自动传递给子生成器）
  3. 子生成器的 return 值通过 StopIteration.value 返回给委托方

任务：
  1. 补全 flatten：用 yield from 展平任意深度的嵌套列表
  2. 补全 pipeline：用 yield from 串联两个生成器
  3. 补全 delegator：观察子生成器的 return 值如何被捕获

问题思考：
  - yield from 和 for x in sub: yield x 有什么本质区别？
  - 这和 asyncio 里 await 的底层机制有什么关系？
"""

# ── 展平嵌套列表 ──────────────────────────────────────────────────────────────
def flatten(nested):
    """
    flatten([1, [2, [3, 4]], 5]) → 1 2 3 4 5

    TODO: 遍历 nested 的每个元素
      - 如果是列表，yield from flatten(elem)  ← 递归委托
      - 否则，yield elem
    """
    pass

# ── 子生成器的 return 值 ───────────────────────────────────────────────────────
def sub_gen():
    yield 1
    yield 2
    return "sub done"   # return 值通过 StopIteration.value 传递

def delegator():
    # TODO: result = yield from sub_gen()
    #       print(f"  子生成器返回值: {result}")
    #       yield 99   ← 委托完成后自己再 yield 一个值
    pass

# ── yield from vs 手动 for 循环的区别（send 透传）─────────────────────────────
def inner():
    """接收 send 进来的值，加倍后 yield 出去"""
    while True:
        val = yield
        if val is None:
            return
        yield val * 2

def with_yield_from():
    """用 yield from 透传 send，inner 能收到外部 send 的值"""
    # TODO: yield from inner()
    pass

def with_manual_loop():
    """手动 for 循环，send 无法透传到 inner"""
    gen = inner()
    next(gen)
    for val in gen:
        yield val   # inner 里的 yield val*2 出来，但 send 进不去

def main():
    print("── flatten 展平嵌套列表 ──")
    nested = [1, [2, [3, 4]], [5, 6], 7]
    # TODO: 打印 list(flatten(nested))

    print("\n── delegator 捕获子生成器返回值 ──")
    gen = delegator()
    # TODO: 用 for 遍历 gen，打印所有值（应该是 1, 2, 99）

    print("\n── yield from 的 send 透传 ──")
    print("  with_yield_from:")
    gen = with_yield_from()
    next(gen)          # 启动
    # TODO: send 10, 20, 30 进去，打印每次结果（应该是 20, 40, 60）

    print("  with_manual_loop (send 无法透传):")
    gen = inner()
    next(gen)
    try:
        print(f"  send 10 → {gen.send(10)}")   # 能收到
        # 但如果套一层 with_manual_loop，send 进不到 inner
    except StopIteration:
        pass

if __name__ == "__main__":
    main()
