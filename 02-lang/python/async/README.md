# Python 异步编程

## 1. 基础API

asyncio的本质是协程，而使用await可以把控制权交还给事件循环，并注册一个完成后叫醒我的回调。如下所示，两个任务在一个线程上，如果希望启动多个任务，那么使用 `asyncio.gather`

```python
await asyncio.gather(worker("A", 3), worker("B", 3), worker("C", 3))
```

协程的调度是**非抢占式**的：只有 `await` 才会让出控制权，没有任何机制能强行打断一个协程。

但调度顺序**不是**「按传入顺序」。就绪队列 `loop._ready` 是 FIFO，`gather` 的传参顺序只决定**初始**入队顺序；之后每次让出后重新入队，顺序由重新入队的先后决定。而且 `_run_once` 会**先快照队列长度**，只跑本轮开始时已在队列里的 Handle，本轮新入队的要等下一轮：

```python
def _run_once(self):
    ntodo = len(self._ready)      # ★ 先快照
    for i in range(ntodo):
        handle = self._ready.popleft()
        if not handle._cancelled:
            handle._run()
    # 本轮新 call_soon 进来的，等下一轮
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

## 3. await 让出了什么

`asyncio.sleep` 的两个分支走的**不是**同一条路径：

```python
async def sleep(delay, result=None):
    if delay <= 0:
        await __sleep0()      # 分支 A：裸 yield
        return result
    loop = events.get_running_loop()
    future = loop.create_future()
    h = loop.call_later(delay, futures._set_result_unless_cancelled, future, result)
    try:
        return await future   # 分支 B
    finally:
        h.cancel()

async def __sleep0():
    yield                     # 不产生任何 Future
```

|分支|路径|碰到的结构|
|---|---|---|
|`sleep(0)`|`coro.send(None)` 拿到 `None` → `Task.__step` 调 `call_soon(self.__step)`|只有**就绪队列**|
|`sleep(0.1)`|`call_later` 造 TimerHandle 进堆 → 挂起 → 到期弹进就绪队列 → 回调 `set_result` → 触发 `Task.__step`|**定时器堆 → 就绪队列**，两跳|

`sleep(0)` 的准确语义是「**放弃本轮剩余，排到下一轮**」——因为 `Task.__step` 对 `result is None` 的处理就是重新 `call_soon` 自己，排到队尾。

**如果一个协程从头到尾一次 `await` 都没有**：它会在**一次 `__step` 里跑到结束**，独占整个循环，没有抢占机制能打断它。

## 4. asyncio 与 CPU 密集任务

- **asyncio 救不了**：单线程 + 无抢占。CPU 密集协程没有 `await` → 无让出点 → 事件循环整个冻住。这是**调度**问题，GIL 还没上场。
- **多线程也救不了纯 Python 计算**：GIL 保证同一时刻只有一个线程执行**字节码**，纯 Python 循环不提速，还多出切换开销。
- **多进程可以**：独立解释器、独立 GIL、独立内存 → 真并行。代价是进程开销 + IPC 序列化。

### 反转：多线程对 CPU 密集任务是有效的

只要那段计算在 C 扩展里**主动释放了 GIL**：

```python
with ThreadPoolExecutor(8) as ex:
    list(ex.map(lambda i: torch.matmul(a, b), range(8)))   # 真并行
```

`torch.matmul` / NumPy 大部分算子在 C++ 里跑，执行期间释放 GIL，8 个线程在 CPU 上真并行。

所以「GIL 让多线程没用」是过度简化。准确判据是：**这段计算花在 Python 字节码上，还是花在释放了 GIL 的 C 扩展里？** 前者用进程池，后者用线程池。

### free-threading 版本节奏

- 3.12：PEP 684，子解释器可各自持有独立 GIL
- 3.13：PEP 703，实验性 free-threaded 构建（`python3.13t`）
- 3.14：PEP 779，free-threaded 成为官方支持的构建模式，但仍非默认

版本节奏变化快，以实际使用的版本为准。

### 接回 PyTorch DataLoader

`DataLoader(num_workers>0)` 默认用**多进程**：`__getitem__` 里主要是 Python 字节码 + PIL 解码 + 增强，前者不放 GIL；多进程同时绕开了 GIL 和对象共享。代价是 worker 的 fork/spawn 开销，以及要把 batch 搬回主进程——这正是 `pin_memory` + `prefetch_factor` 存在的原因。

