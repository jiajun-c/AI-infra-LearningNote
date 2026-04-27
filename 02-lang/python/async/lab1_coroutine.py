"""
asyncio Lab 1：协程的本质 —— 它不是线程
===========================================
目标：理解 coroutine 是"可暂停的函数"，await 是主动让出控制权的唯一方式。

任务：
  1. 补全 task_a / task_b，在每个步骤前后打印时间戳和当前任务名
  2. 运行后观察输出顺序：两个任务是交替执行的，还是串行的？
  3. 把 asyncio.sleep 换成 time.sleep，再观察输出有何变化，解释原因
"""

import asyncio
import time

def ts():
    return f"{time.perf_counter():.3f}s"

async def task_a():
    print(f"[{ts()}] A: 开始")
    await asyncio.sleep(1)
    print("A 醒来")
    await asyncio.sleep(1)
    print("A 结束")
    pass

async def task_b():
    print(f"[{ts()}] B: 开始")
    await asyncio.sleep(0.5)
    print("B 醒来")
    await asyncio.sleep(0.5)
    print("B 结束")
    pass

async def main():
    await asyncio.gather(task_a(), task_b())
    pass

if __name__ == "__main__":
    t0 = time.perf_counter()
    asyncio.run(main())
    print(f"总耗时: {time.perf_counter() - t0:.3f}s")
