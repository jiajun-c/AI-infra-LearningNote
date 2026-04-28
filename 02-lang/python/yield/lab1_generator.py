"""
yield Lab 1：生成器的本质 —— 可暂停的函数
===========================================
目标：理解 yield 是一个"暂停点"，函数执行到 yield 时挂起，
      调用方用 next() 推进，每次 next() 从上次 yield 处继续。

任务：
  1. 补全 counter：每次 yield 一个数，从 start 到 end-1
  2. 用 next() 手动推进，观察 StopIteration
  3. 用 for 循环遍历，对比两种用法
  4. 在 yield 前后打印时间戳，验证"暂停"行为

问题思考：
  - counter() 调用后立刻执行了吗？
  - 生成器对象和列表的内存占用有何差异？
"""

import time

def ts():
    return f"{time.perf_counter():.3f}s"

def counter(start: int, end: int):
    print(f"  [{ts()}] 生成器启动")
    for i in range(start, end):
        print(f"  [{ts()}] 即将 yield {i}")
        yield i
        print(f"  [{ts()}] yield {i} 返回，继续执行")
    print(f"  [{ts()}] 生成器结束")

def main():
    print("── 调用 counter()，此时函数体执行了吗？ ──")
    gen = counter(0, 4)       # 只是创建生成器对象，函数体不执行
    print(f"  gen = {gen}")   # <generator object ...>

    print("\n── 手动 next() 推进 ──")
    for i in range(5):
        try:
            next(gen)
        except StopIteration:
            print("StopIteration")
        finally:
            pass
            
    # TODO: 调用 next(gen) 四次，每次打印返回值
    # 第五次调用 next(gen) 会抛 StopIteration，用 try/except 捕获并打印

    print("\n── for 循环遍历（自动处理 StopIteration）──")
    # TODO: for v in counter(0, 4): print(f"  got {v}")
    for v in counter(0, 4):
        print(f"got {v}")
    print("\n── 内存对比 ──")
    import sys
    n = 1_000_000
    lst = list(range(n))
    gen = (x for x in range(n))   # 生成器表达式
    print(f"  list(range({n})): {sys.getsizeof(lst):,} bytes")
    print(f"  generator expr:   {sys.getsizeof(gen):,} bytes")

if __name__ == "__main__":
    main()
