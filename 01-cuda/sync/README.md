# 同步机制

CUDA 中的异步同步与流水线机制。

## 子目录

| 目录 | 内容 |
|------|------|
| [pipe/](pipe/README.md) | `cuda::pipeline` API — 生产者-消费者流水线 |
| [mem/](mem/README.md) | `memcpy_async` + `cp.async` — 异步内存拷贝 |
| [stream/](stream/) | Stream 级别的同步与 overlap |

## 同步层级

```
Grid ← Cooperative Groups grid_group
  Cluster ← cluster_group (Hopper+, mbarrier)
    Block ← thread_block (__syncthreads)
      Warp ← coalesced_group (warp shuffle)
```

### Pipeline 模式

```
Stage 0: [load GMEM→SMEM] → [compute SMEM→REG] → [store REG→GMEM]
Stage 1:                    [load GMEM→SMEM] → [compute...]
Stage 2:                                       [load...]
         ←─── 时间 ───→
```

## 相关

- [Cooperative Groups](../cg/README.md)
- [Ampere cp.async](../ampere/cpasync/README.md)
- [Hopper Pipeline](../hopper/pipe/README.md) — CUTLASS 内部 Pipeline 实现
- [Ampere cp.async](../ampere/cpasync/README.md)
- [Hopper Pipeline](../hopper/pipe/README.md)
