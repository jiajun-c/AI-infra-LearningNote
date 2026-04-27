# Python 异步编程

## 1. 基础API

asyncio的本质是协程，而使用await可以把控制权交还给事件循环，并注册一个完成后叫醒我的回调。如下所示，两个任务在一个线程上，如果希望启动多个任务，那么使用 `asyncio.gather` 

携程的调度策略是非抢占式的，其按照传入的顺序来进行调度，当发生await后也会按照顺序去调度到下一个

```python
await asyncio.gather(worker("A", 3), worker("B", 3), worker("C", 3))
```


```python
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

```

## 2. 队列接口

使用 async.Queue，这是一个线程安全的FIFO队列，接口分为四类

```shell
await queue.put(item)    # 放入一个 item，队满时挂起直到有空位
await queue.get()        # 取出一个 item，队空时挂起直到有 item

queue.put_nowait(item)   # 立即放入，队满时抛 QueueFull（不挂起）
queue.get_nowait()       # 立即取出，队空时抛 QueueEmpty（不挂起）
```

完成通知，消费者方发起 `task_done`，队列那边通过 `queue.join()` 等待全部任务完成

```shell
queue.task_done()        # 消费者处理完一个 item 后调用，内部计数器 -1
await queue.join()       # 阻塞直到所有已入队的 item 都被 task_done() 标记
```

