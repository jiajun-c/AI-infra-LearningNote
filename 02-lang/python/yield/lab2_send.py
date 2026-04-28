"""
yield Lab 2：send() —— 双向通道
=================================
目标：理解 yield 不只是输出，也可以接收值。
      send(value) 把 value 传给 yield 表达式，同时推进生成器。

      val = yield output
            ↑             ↑
      接收 send 的值    向外输出的值

任务：
  1. 补全 accumulator：每次 send 一个数字进来，yield 出累计和
  2. 注意：第一次必须 send(None)（或 next()）来启动生成器
  3. 补全 echo_upper：接收字符串，yield 出大写版本

问题思考：
  - 为什么第一次不能 send 非 None 值？
  - yield 表达式的值（左边）和 yield 的值（右边）分别是什么？
"""

def accumulator():
    """
    用法：
        gen = accumulator()
        next(gen)          # 启动，推进到第一个 yield
        gen.send(10)       # → 10
        gen.send(20)       # → 30
        gen.send(5)        # → 35
    """
    total = 0
    while True:
        # TODO: received = yield total
        #       total += received
        received = yield total
        total += received
        pass

def echo_upper():
    text = None
    while True:
        text = yield text.upper() if text else None


def main():
    print("── accumulator ──")
    gen = accumulator()
    next(gen)          # 启动生成器，推进到第一个 yield
    # TODO: 依次 send 10, 20, 5，打印每次的返回值
    print(gen.send(10))
    print(gen.send(20))
    print(gen.send(5))
    
    print("\n── echo_upper ──")
    gen = echo_upper()
    next(gen)
    # TODO: 依次 send "hello", "world", "asyncio"，打印每次返回值
    print(gen.send("hello"))
    print(gen.send("world"))
    print("\n── 第一次 send 非 None 会怎样？ ──")
    gen2 = accumulator()
    try:
        gen2.send(99)   # 抛 TypeError：can't send non-None value to a just-started generator
    except TypeError as e:
        print(f"  TypeError: {e}")

if __name__ == "__main__":
    main()
