"""
asyncio Lab 5：生产者-消费者 与 asyncio.Queue
===============================================
目标：用 asyncio.Queue 实现生产者-消费者，理解背压（backpressure）机制。
      这是 AI 推理服务中请求调度的核心模式（vLLM scheduler 的简化版）。

背景：
  asyncio.Queue(maxsize=N) 是有界队列：
    - put() 在队满时自动挂起生产者（背压），直到消费者取走一个
    - get() 在队空时自动挂起消费者，直到生产者放入一个
  这避免了生产者把消费者淹没（OOM），是流控的基础原语。

任务：
  1. 补全 producer：生成 total 个"请求"，每个请求随机耗时 0.1~0.5s
     把请求放入队列，队满时自动等待（体验背压）
  2. 补全 consumer：从队列取请求，模拟处理（sleep），打印处理结果
  3. 补全 main：启动 1 个生产者 + 3 个消费者，等待所有请求处理完毕
     提示：用 queue.join() 等待所有 item 被 task_done() 标记

  4. 扩展（选做）：改为 2 个生产者 + 5 个消费者，观察吞吐变化

问题思考：
  - maxsize=1 和 maxsize=100 对系统行为有何影响？
  - 如果消费者抛异常没有调用 task_done()，join() 会发生什么？
  - vLLM 中 scheduler 如何用类似机制控制 GPU 显存使用？
"""

import asyncio
import random
import time

async def producer(queue: asyncio.Queue, producer_id: int, total: int):
    for i in range(total):
        delay = random.uniform(0.1, 0.5)
        item  = {"id": f"P{producer_id}-{i}", "delay": delay}
        # 队满时 put 自动挂起，直到消费者取走一个（背压）
        await queue.put(item)
        print(f"  [生产者 {producer_id}] 生产: {item['id']}  队列深度: {queue.qsize()}")
    print(f"[生产者 {producer_id}] 全部生产完毕")

async def consumer(queue: asyncio.Queue, consumer_id: int):
    while True:
        try:
            item = await queue.get()
            await asyncio.sleep(item["delay"])
            print(f"  [消费者 {consumer_id}] 处理完 {item['id']}  耗时: {item['delay']:.2f}s")
            queue.task_done()   # 必须调用，否则 queue.join() 永远不返回
        except asyncio.CancelledError:
            # 外部 cancel 时优雅退出，不能在这里 task_done（item 可能未处理完）
            break

async def main():
    N_PRODUCERS = 1
    N_CONSUMERS = 3
    ITEMS_PER_PRODUCER = 10
    QUEUE_SIZE = 5   # 有界队列，体验背压

    queue = asyncio.Queue(maxsize=QUEUE_SIZE)
    print(f"队列大小: {QUEUE_SIZE}，生产者: {N_PRODUCERS}，消费者: {N_CONSUMERS}")

    t0 = time.perf_counter()

    # 先启动消费者（无限循环，等待 item）
    consumer_tasks = [
        asyncio.create_task(consumer(queue, i))
        for i in range(N_CONSUMERS)
    ]

    # 启动生产者，等待全部生产完毕
    producer_tasks = [
        asyncio.create_task(producer(queue, i, ITEMS_PER_PRODUCER))
        for i in range(N_PRODUCERS)
    ]
    await asyncio.gather(*producer_tasks)

    # 等待队列中所有 item 被消费者 task_done() 标记
    await queue.join()

    # 消费者是无限循环，手动 cancel
    for t in consumer_tasks:
        t.cancel()
    await asyncio.gather(*consumer_tasks, return_exceptions=True)

    print(f"\n总耗时: {time.perf_counter() - t0:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
