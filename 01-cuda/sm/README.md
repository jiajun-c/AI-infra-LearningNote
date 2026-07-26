# SM 实验

SM（Streaming Multiprocessor）级别的性能实验与分析。

## 子目录

| 目录 | 内容 | 关键发现 |
|------|------|---------|
| [sm_interval/](sm_interval/) | H100 L2 缓存分区分析 | SM 放置策略影响 L2 命中率和带宽 |
| [gemv/](gemv/) | GEMV 实现与 L2 延迟测试 | 多种 GEMV 变体、cuBLAS 基准对比 |
| [libsmctrl/](libsmctrl/) | SM 掩码控制实验 | H100 SM ID 与物理 TPC 不一一对应 |
| [element_wise/](element_wise/) | 逐元素内核测试 | — |
| [computeMemory/](computeMemory/) | SM 计算与内存测试 | — |
| [device/](device/) | 设备信息查询 | — |
| [green_context/](green_context/) | Green Context 实验 | — |

## 核心结论

1. **L2 分区竞争**：相邻 SM 可能映射到同一 L2 partition，SM 放置策略对中等数据量的 kernel 影响显著
2. **SM 掩码映射**：H100 的物理 TPC 单元与 SM ID 的映射关系需要校准，不能简单假设线性对应
3. **GEMV 瓶颈**：GEMV 是典型的 memory-bound 操作，L2 缓存行为决定性能上限

## 相关文档

- [SM 微架构](../hardware/sm.md)
- [H100 缓存分析](sm_interval/h100_cache_analysis.md)
