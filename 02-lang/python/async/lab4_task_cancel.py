"""
asyncio Lab 4：Task、取消与超时
==================================
目标：理解 asyncio.Task 的生命周期管理，掌握超时和取消机制。

背景：
  asyncio.create_task() 把协程包装成 Task 立刻调度（不用 await 也会运行）。
  Task 可以被取消（cancel），被取消的 Task 会在下一个 await 点抛出 CancelledError。
  asyncio.wait_for() 给单个协程加超时；asyncio.timeout() 是 3.11+ 的上下文管理器写法。

任务：
  1. 补全 slow_job：每秒打印进度，共运行 n 秒
  2. 补全 lab_cancel：启动 slow_job(5)，2 秒后手动 cancel，捕获 CancelledError
  3. 补全 lab_timeout：用 asyncio.wait_for 给 slow_job(5) 加 2 秒超时，捕获 TimeoutError
  4. 补全 lab_shield：用 asyncio.shield 保护 slow_job(3) 不被外部取消，观察区别

问题思考：
  - cancel() 和 wait_for 超时的区别是什么？
  - shield 保护的任务在外部 task 被取消后是否还会继续运行？
"""

import asyncio
import time

async def slow_job(name: str, n: int):
    try:
        for i in range(n):
            print(f"  [{name}] step {i+1}/{n}")
            # TODO: await asyncio.sleep(1)
            await asyncio.sleep(1)
            pass
        print(f"  [{name}] 完成")
        return f"{name}:done"
    except asyncio.CancelledError:
        print(f"  [{name}] 被取消!")
        raise   # 必须重新 raise，否则取消语义丢失

async def lab_cancel():
    print("\n── cancel ──")
    task = asyncio.create_task(slow_job("cancel", 5))
    await asyncio.sleep(2)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("  lab_cancel: 捕获到 CancelledError，任务已取消")

async def lab_timeout():
    print("\n── wait_for timeout ──")
    try:
        await asyncio.wait_for(slow_job("timeout", 5), timeout=2.0)
    except asyncio.TimeoutError:
        print("  lab_timeout: 捕获到 TimeoutError，超时取消")

async def lab_shield():
    print("\n── shield ──")
    inner = asyncio.create_task(slow_job("shielded", 3))
    # shield 包装后的 Future 被取消，不会传播到 inner
    try:
        await asyncio.wait_for(asyncio.shield(inner), timeout=1.0)
    except asyncio.TimeoutError:
        print("  lab_shield: 外部超时，但 inner 仍在运行")
    # inner 没有被取消，继续等待它正常完成
    result = await inner
    print(f"  lab_shield: inner 最终结果 = {result}")

async def main():
    await lab_cancel()
    await lab_timeout()
    await lab_shield()

if __name__ == "__main__":
    asyncio.run(main())
