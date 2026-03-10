# 连续批处理 (Continuous Batching)

连续批处理（Continuous Batching）是一种针对 LLM 推理服务的高效调度技术，用于在请求到达时间不一致的情况下最大化 GPU 利用率。

## 背景问题

在实际的 LLM 推理服务场景中，用户的请求是**异步到达**的：

```
时间轴：
t₀: 请求 A 到达 ──┐
t₁: 请求 B 到达 ──┼── 请求处理中...
t₂: 请求 C 到达 ──┘
```

### 传统静态批处理的问题

静态批处理（Static Batching）需要等待批次中**所有请求都完成**才能释放资源：

```
┌────────────────────────────────────────────────────┐
│ Batch: [A, B, C]                                   │
│                                                    │
│ A: [████████████████████████████████] done at t₁₀  │
│ B: [████████████] done at t₅  ← 完成但无法释放      │
│ C: [████████████████] done at t₇  ← 完成但无法释放 │
│                                                    │
│ GPU 利用率低：B、C 完成后资源仍被占用               │
└────────────────────────────────────────────────────┘
```

**问题**：
- 短请求被长请求阻塞
- 已完成的请求无法及时释放显存
- 新请求需要等待当前批次全部完成才能加入
- GPU 利用率低，吞吐量受限

## 连续批处理的核心思想

**当一个请求完成时，立即从批次中移除，并将新请求插入到当前批次中继续执行。**

```
┌────────────────────────────────────────────────────┐
│ 连续批处理执行流程                                  │
│                                                    │
│ iter 1-5:  [A, B, C]  ← 初始批次                    │
│ iter 6-10: [A, C, D]  ← B 完成，D 加入              │
│ iter 11-15:[A, D, E]  ← C 完成，E 加入              │
│ iter 16-20:[D, E, F]  ← A 完成，F 加入              │
│ ...                                                │
└────────────────────────────────────────────────────┘
```

### 关键特性

1. **细粒度调度**：以 iteration 为单位进行调度决策
2. **动态插入**：新请求可立即加入正在执行的批次
3. **即时释放**：请求完成后立即释放显存和计算资源
4. **高吞吐量**：持续保持 GPU 高利用率

## 与相关技术的对比

| 技术 | 批处理粒度 | 请求加入时机 | 显存管理 |
|------|-----------|-------------|---------|
| 静态批处理 | 批次级 | 批次开始前 | 批次完成后释放 |
| 动态批处理 | 批次级 | 批次间等待 | 批次完成后释放 |
| **连续批处理** | **iteration 级** | **任意 iteration** | **请求完成即释放** |

## 与 PagedAttention 的结合

连续批处理通常与 **PagedAttention** 结合使用，实现高效的显存管理：

```
┌─────────────────────────────────────────────────────┐
│              PagedAttention + Continuous Batching   │
│                                                     │
│  请求 A (生成长) → Block Table: [0x01, 0x05, 0x0A]  │
│  请求 B (刚加入) → Block Table: [0x02]              │
│  请求 C (生成中) → Block Table: [0x03, 0x07]        │
│                                                     │
│  非连续物理块 → 支持动态分配/释放                    │
└─────────────────────────────────────────────────────┘
```

**优势**：
- 显存按需分配，无碎片化
- 支持任意数量的并发请求
- 请求完成时显存立即回收

## 调度算法

### 基本流程

```python
def continuous_batching_scheduler(request_queue, running_requests, max_batch_size):
    """
    连续批处理调度器
    """
    while True:
        # 1. 执行一次前向传播
        outputs = model.forward(running_requests)

        # 2. 检查已完成的请求
        completed = []
        for req in running_requests:
            if req.is_done():
                completed.append(req)
                req.release_resources()  # 释放 KV Cache

        # 3. 从运行队列中移除已完成的请求
        for req in completed:
            running_requests.remove(req)

        # 4. 尝试将新请求加入批次
        while len(running_requests) < max_batch_size and request_queue:
            new_req = request_queue.pop(0)
            if can_fit_in_memory(new_req):  # 检查显存是否足够
                running_requests.append(new_req)
                new_req.allocate_resources()

        # 5. 继续下一个 iteration
        yield running_requests
```

### 调度策略

| 策略 | 描述 | 适用场景 |
|------|------|---------|
| FCFS (First-Come-First-Served) | 按到达顺序处理 | 通用场景 |
| Priority-Based | 优先处理高优先级请求 | 多租户服务 |
| Length-Aware | 优先处理短请求 | 降低平均延迟 |
| Deadline-Aware | 考虑截止时间 | SLA 敏感场景 |

## 性能指标

### 吞吐量提升

假设平均请求长度为 L，请求到达间隔为 Δt：

| 批处理大小 | 静态批处理吞吐量 | 连续批处理吞吐量 |
|-----------|-----------------|-----------------|
| 1 | 1/L | 1/L |
| 4 | ~4/(L_max) | ~4/L_avg |
| 8 | ~8/(L_max) | ~8/L_avg |

> 连续批处理的吞吐量接近理想值，而静态批处理受最长请求限制

### 延迟对比

```
请求到达时间：t=0, 1, 2, 3, 4
请求生成长度：10, 5, 8, 3, 6 tokens

静态批处理 (batch=4):
- 请求 A: 等待 0 + 执行 10 = 10
- 请求 B: 等待 0 + 执行 10 = 10  ← 被 A 阻塞
- 请求 C: 等待 0 + 执行 10 = 10  ← 被 A 阻塞
- 请求 D: 等待 0 + 执行 10 = 10  ← 被 A 阻塞
- 请求 E: 等待 10 + 执行 6 = 16 ← 等待整个批次

连续批处理:
- 请求 A: 执行 10 (t=0-10)
- 请求 B: 执行 5 (t=1-6)   ← 完成后立即释放
- 请求 C: 执行 8 (t=2-10)
- 请求 D: 执行 3 (t=3-6)   ← 完成后立即释放
- 请求 E: 执行 6 (t=4-10)
```

## 实现示例

以下是简化的连续批处理实现：

```python
import torch
from typing import List, Optional

class Request:
    """单个推理请求"""
    def __init__(self, request_id: str, prompt_ids: List[int], max_tokens: int):
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.max_tokens = max_tokens
        self.generated_tokens = []
        self.kv_cache = None  # Paged KV Cache

    def is_done(self) -> bool:
        return len(self.generated_tokens) >= self.max_tokens

    def allocate_kv_cache(self, block_size: int, num_blocks: int):
        """分配 KV Cache"""
        self.kv_cache = torch.randn(
            num_blocks, block_size,
            num_layers, num_heads, head_dim
        )

    def free_kv_cache(self):
        """释放 KV Cache"""
        if self.kv_cache is not None:
            del self.kv_cache
            self.kv_cache = None


class ContinuousBatcher:
    """连续批处理调度器"""
    def __init__(self, model, max_batch_size: int, max_memory: float):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_memory = max_memory
        self.running_requests: List[Request] = []
        self.pending_queue: List[Request] = []

    def add_request(self, request: Request):
        """添加新请求"""
        self.pending_queue.append(request)

    def step(self) -> List[Request]:
        """执行一个 iteration，返回已完成的请求"""
        completed = []

        if not self.running_requests:
            return completed

        # 1. 执行前向传播
        outputs = self.model.forward(self.running_requests)

        # 2. 处理输出，更新请求状态
        for i, request in enumerate(self.running_requests):
            token = outputs[i].argmax(dim=-1)
            request.generated_tokens.append(token.item())

            # 检查是否完成
            if request.is_done():
                completed.append(request)

        # 3. 移除已完成的请求
        for request in completed:
            self.running_requests.remove(request)
            request.free_kv_cache()

        # 4. 尝试加入新请求
        self._schedule_pending()

        return completed

    def _schedule_pending(self):
        """调度等待队列中的请求"""
        while self.pending_queue and len(self.running_requests) < self.max_batch_size:
            request = self.pending_queue[0]

            # 检查是否有足够显存
            required_mem = self._estimate_memory(request)
            available_mem = self.max_memory - self._current_memory_usage()

            if required_mem <= available_mem:
                self.pending_queue.pop(0)
                self.running_requests.append(request)
                request.allocate_kv_cache(...)
            else:
                break  # 显存不足，等待
```

## 实际应用

### vLLM 中的 Continuous Batching

vLLM 是首个广泛使用的 Continuous Batching 实现：

```
vLLM 架构:

┌─────────────────────────────────────────┐
│            Scheduler                     │
│  - 维护 running / waiting 队列           │
│  - 每 step 检查完成情况                   │
│  - 动态插入新请求                        │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│         PagedAttention                   │
│  - 非连续 KV Cache                       │
│  - 按需分配/释放                         │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│           GPU Executor                   │
│  - 并行执行 attention                    │
│  - 支持变长序列                          │
└─────────────────────────────────────────┘
```

### 关键配置参数

| 参数 | 描述 | 典型值 |
|------|------|-------|
| `max_batch_size` | 最大并发请求数 | 256-2048 |
| `block_size` | PagedAttention 块大小 | 16-64 |
| `gpu_memory_utilization` | GPU 显存使用比例 | 0.9-0.95 |
| `swap_space` | CPU 交换空间大小 | 4-8 GB |

## 优缺点分析

### 优点

1. **高吞吐量**：GPU 持续高效利用，无空闲等待
2. **低延迟**：新请求无需等待批次完成
3. **公平性**：短请求不会被长请求过度阻塞
4. **显存高效**：配合 PagedAttention，无碎片化

### 挑战

1. **调度开销**：每 iteration 需要检查和调度
2. **变长序列**：需要处理不同长度的序列并行
3. **显存管理**：动态分配/释放需要精细控制
4. **同步复杂**：多请求状态管理复杂

## 相关技术

- **PagedAttention**: 分页注意力机制，支持动态显存管理
- **Prefix Cache**: 前缀缓存，优化相同 prompt 的请求
- **Chunked Prefill**: 分块预填充，减少长 prompt 显存占用
- **Speculative Decoding**: 推测解码，进一步加速生成

## 参考资料

- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [Continuous Batching Blog](https://www.anyscale.com/blog/continuous-batching)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
