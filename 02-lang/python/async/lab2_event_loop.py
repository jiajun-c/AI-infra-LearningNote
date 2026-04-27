"""
asyncio Lab 2：事件循环的调度机制
===================================
目标：理解 await asyncio.sleep(0) 的语义，以及任务调度顺序。

背景：
  asyncio 是单线程的，同一时刻只有一个协程在运行。
  await 把控制权交还给事件循环，事件循环决定下一个运行谁。
  await asyncio.sleep(0) 是"主动让出但立刻重新排队"。

任务：
  1. 补全三个 worker，每个 worker 循环 3 次，每次打印"worker X step Y"后 yield
  2. 预测输出顺序后再运行，验证你的预测
  3. 改变 sleep(0) 为 sleep(0.1)，观察顺序是否改变

问题思考：
  - 为什么不用 await 就无法切换到其他协程？
  - CPU 密集任务（纯计算循环）能否用 asyncio 并发？
"""

import asyncio

async def worker(name: str, steps: int):
    for i in range(steps):
        print(f"  {name} step {i}")
        await asyncio.sleep(0.1)
        # TODO: await asyncio.sleep(0)  ← 主动让出，让其他 worker 有机会运行
        pass

async def main():
    await asyncio.gather(worker("A", 3), worker("B", 3), worker("C", 3))
    # TODO: 同时创建 worker("A",3) / worker("B",3) / worker("C",3)
    # 用 asyncio.gather 并发运行
    pass

if __name__ == "__main__":
    asyncio.run(main())
