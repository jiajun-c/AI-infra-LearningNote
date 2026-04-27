"""
asyncio Lab 3：真实 I/O 并发 —— 模拟 HTTP 并发请求
=====================================================
目标：用 asyncio 实现真正的 I/O 并发，理解为什么 asyncio 适合 I/O 密集场景。

背景：
  网络请求的大部分时间在等待服务器响应（I/O 等待），CPU 几乎空闲。
  同步版本：串行等待，总时间 = 所有请求延迟之和
  asyncio版本：并发等待，总时间 ≈ 最长单个请求延迟

  本 lab 用 asyncio.sleep 模拟网络延迟，原理完全相同。
  真实场景用 aiohttp / httpx 替换 fake_http_get 即可。

任务：
  1. 补全 fetch_one：模拟一个耗时 delay 秒的 HTTP GET，返回 f"resp:{url}"
  2. 补全 fetch_all_sequential：串行逐个 await，记录总耗时
  3. 补全 fetch_all_concurrent：用 asyncio.gather 并发，记录总耗时
  4. 对比两种耗时，验证并发版本总时间 ≈ max(delays)
"""

import asyncio
import time

REQUESTS = [
    ("https://api.example.com/a", 1.0),
    ("https://api.example.com/b", 0.5),
    ("https://api.example.com/c", 1.5),
    ("https://api.example.com/d", 0.8),
    ("https://api.example.com/e", 1.2),
]

async def fake_http_get(url: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"resp:{url}"

async def fetch_all_sequential():
    print("\n── 串行 ──")
    t0 = time.perf_counter()
    results = []
    for url, delay in REQUESTS:
        # TODO: 逐个 await fake_http_get，打印每个结果
        await fake_http_get(url, delay)
        # pass
    print(f"串行总耗时: {time.perf_counter() - t0:.2f}s")
    return results

async def fetch_all_concurrent():
    print("\n── 并发 ──")
    t0 = time.perf_counter()
    results = await asyncio.gather(
        *[fake_http_get(url, delay) for url, delay in REQUESTS]
    )
    for r in results:
        print(f"  {r}")
    print(f"并发总耗时: {time.perf_counter() - t0:.2f}s")
    return results

async def main():
    await fetch_all_sequential()
    await fetch_all_concurrent()

if __name__ == "__main__":
    asyncio.run(main())
