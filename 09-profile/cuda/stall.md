# Warp Stall 分析

## 什么是 Warp Stall

GPU 通过**细粒度多线程（fine-grained multithreading）**隐藏延迟——当 warp 0 等待内存数据时，warp scheduler 立即切换到 warp 1。但如果所有 warp 都拿不到所需资源，就进入 **stall**（停顿）状态。

Stall 按持续时间分为两类：

## Short Stall vs Long Stall

| | **Short Stall** | **Long Stall** |
|---|---|---|
| **持续时间** | 几个到十几个 cycle | 几十到几百个 cycle |
| **硬件计数器** (Nsight Compute) | `smsp__warp_issue_stalled_short_scoreboard` | `smsp__warp_issue_stalled_long_scoreboard` |
| **典型原因** | 寄存器依赖、指令缓存缺失、流水线气泡 | 全局内存/HBM 访问延迟、同步屏障、常量内存冲突 |
| **反映的瓶颈** | 指令级/流水线效率问题 | **内存带宽受限** |
| **优化方向** | 减少依赖链、展开循环、调整指令调度 | 减少 HBM 访问、提升 SMEM 复用、coalesced access |

## Nsight Compute 中的 Stall 原因分类

在 **Warp State Statistics** 面板中，主要 stall 类型：

| 状态 | 含义 | 如何解读 |
|------|------|---------|
| **Stalled Short Scoreboard** | 等待之前发射的指令结果（流水线内的短依赖） | 占比高 → 计算瓶颈，优化指令延迟 |
| **Stalled Long Scoreboard** | 等待全局内存/L2/TMU 的数据返回 | 占比高 → **访存瓶颈**，优化内存访问 |
| **Stalled Wait** | 在 `__syncthreads()` 等 barrier 上等待其他 warp | 可能是负载不均或过度同步 |
| **Stalled Not Selected** | warp 准备好了但 scheduler 没选它 | **反而是好事**——说明有足够多的 warp 可以调度，高 occupancy 的标志 |
| **Stalled MIO Throttle** | MIO（内存 IO）指令队列满 | 访存指令过多，流水线堵住 |
| **Stalled Math** | math 执行单元繁忙 | kernel 确实是计算密集 |

## 与算术强度的对应关系

[理论分析](./theory.md) 中计算出的算术强度（AI），在 profiler 中的表现：

```text
AI < 拐点 → 理论上访存瓶颈 → 表现为 Long Stall / MIO Throttle 占比高
AI > 拐点 → 理论上计算瓶颈 → 表现为 Short Scoreboard / Math Stall 占比高
```

例如 H100 上 online softmax（AI = 2.5，拐点 = 153），profiler 中预期看到：
- **Long Scoreboard** 占比显著高于 Short Scoreboard
- 优化应聚焦于减少 HBM 访存（如 kernel fusion，用 SMEM 缓存）

## 分析流程

1. 用 Nsight Compute 采集 kernel 的 warp state 统计
2. 找到占比最高的 stall 原因
3. 对照 [理论分析](./theory.md) 中的 roofline 判断
4. 针对瓶颈类型选择优化方向：
   - 访存瓶颈 → 增大 AI（fusion、tiling、SMEM）
   - 计算瓶颈 → 提升计算效率（TensorCore、减少依赖链、提高 occupancy）
