"""
yield Lab 5：用生成器手写一个迷你 asyncio
==========================================
目标：理解 asyncio 的事件循环本质上就是一个驱动生成器的调度器。
      这个 lab 用纯 yield（不用 async/await）实现并发调度，
      揭示 await 底层的 yield from 机制。

背景：
  Python 3.4 之前的 asyncio 就是用 @asyncio.coroutine + yield from 实现的。
  async/await 只是语法糖，底层机制完全相同：
    await x  ≡  yield from x.__await__()

任务：
  1. 理解 sleep_gen：产出一个 Future，事件循环看到 Future 后注册定时器
  2. 补全 task_a / task_b：用 yield from 模拟 await
  3. 补全 run_loop：最简单的事件循环，驱动所有协程交替运行

实验步骤：
  - 先读懂框架代码，理解 Future/Task 的最小实现
  - 补全 TODO 部分，运行并观察输出顺序
"""

import time
import heapq

# ── 最小 Future：持有一个未来的值 ────────────────────────────────────────────
class Future:
    def __init__(self):
        self._result = None
        self._done = False
        self._callbacks = []

    def set_result(self, result):
        self._result = result
        self._done = True
        for cb in self._callbacks:
            cb(self)

    def add_done_callback(self, cb):
        if self._done:
            cb(self)
        else:
            self._callbacks.append(cb)

    def __iter__(self):
        if not self._done:
            yield self          # 把自己产出给事件循环，让循环等我完成
        return self._result     # 完成后 return，值通过 StopIteration.value 传递

# ── 定时器堆：(触发时间, future) ─────────────────────────────────────────────
_timers = []

def sleep_gen(seconds: float):
    """yield from sleep_gen(1) ≡ await asyncio.sleep(1)"""
    future = Future()
    deadline = time.perf_counter() + seconds
    heapq.heappush(_timers, (deadline, future))
    yield from future       # 把 future 产出给事件循环，等定时器触发

# ── 协程（用生成器实现）────────────────────────────────────────────────────────
def task_a():
    print(f"  [A] 开始 {time.perf_counter():.2f}s")
    # TODO: yield from sleep_gen(1.0)
    print(f"  [A] 醒来 {time.perf_counter():.2f}s")
    # TODO: yield from sleep_gen(1.0)
    print(f"  [A] 结束 {time.perf_counter():.2f}s")

def task_b():
    print(f"  [B] 开始 {time.perf_counter():.2f}s")
    # TODO: yield from sleep_gen(0.5)
    print(f"  [B] 醒来 {time.perf_counter():.2f}s")
    # TODO: yield from sleep_gen(0.5)
    print(f"  [B] 结束 {time.perf_counter():.2f}s")

# ── 最小事件循环 ───────────────────────────────────────────────────────────────
def run_loop(*coros):
    """
    驱动多个生成器协程并发运行。

    TODO：补全事件循环主体
      ready = [生成器对象列表]
      while ready or _timers:
          1. 驱动所有 ready 中的协程，调用 next(coro)
             - 如果产出一个 Future，说明协程在等待，暂不放回 ready
             - 如果 StopIteration，协程结束，移除
          2. 检查 _timers 堆，找出已到期的定时器，set_result() 唤醒对应 Future
             - Future 完成后，等待它的协程重新加入 ready
          3. 如果 ready 为空但 _timers 非空，time.sleep 到最近的定时器
    """
    # 把协程函数转成生成器对象
    tasks = [coro() for coro in coros]

    # TODO: 实现事件循环
    pass

if __name__ == "__main__":
    t0 = time.perf_counter()
    run_loop(task_a, task_b)
    print(f"\n总耗时: {time.perf_counter() - t0:.2f}s  (预期 ~2s)")
