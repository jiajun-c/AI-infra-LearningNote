# GPU 性能分析

针对 CUDA kernel 的性能分析方法，从打标记定位热点、理论瓶颈分析，到硬件计数器 stall 诊断。

## 内容索引

| 文件 | 主题 | 说明 |
| ---- | ---- | ---- |
| [nvtx.md](./nvtx.md) | nvtx 性能标记 | 在代码中添加 range annotation，配合 Nsight Systems 定位热点 |
| [theory.md](./theory.md) | 理论性能分析 | GFlops 计算、算术强度（AI）、Roofline 拐点、kernel launch 与 wave 效应 |
| [stall.md](./stall.md) | Warp Stall 分析 | Short/Long Stall 的含义，Nsight Compute 计数器解读，与 AI 的对应关系 |
| [roofline.md](./roofline.md)| roofline 分析| 对roofline的分析 |

## 典型分析流程

1. **定位热点**：用 [nvtx](./nvtx.md) 标记代码段，Nsight Systems 找到耗时最长的 kernel
2. **理论估算**：用 [理论分析](./theory.md) 计算 kernel 的算术强度，判断是 compute-bound 还是 memory-bound
3. **硬件验证**：用 [Warp Stall 分析](./stall.md) 查看 Nsight Compute 的 warp state 统计，验证理论判断
4. **针对性优化**：访存瓶颈 → fusion/tiling/SMEM；计算瓶颈 → TensorCore/减少依赖/提高 occupancy
