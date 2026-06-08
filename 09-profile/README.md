# 性能分析与调试

GPU 程序性能分析、调试和优化方法。

## 子目录

| 目录 | 说明 |
|------|------|
| [cuda/](./cuda/README.md) | CUDA kernel 性能分析：nvtx 标记、理论 GFlops 计算、Warp Stall 分析 |
| [debug/](./debug/README.md) | 调试基础：race condition 检测、常见 bug 定位 |
| [improve/](./improve/README.md) | 性能优化方法：定位瓶颈、优化策略 |
| [perplexity/](./perplexity/README.md) | 模型困惑度分析与评估 |
| [thop/](./thop/README.md) | THOP：PyTorch 模型 FLOPs/参数量统计工具 |
| latency/ | GPU kernel 延迟测试（latency_test.cu） |
| log/ | PyTorch profiler trace 日志（chrome trace 格式） |

## 快速入口

- 计算 kernel FLOPs → [cuda/theory.md](./cuda/theory.md)
- 定位 kernel 瓶颈是计算还是访存 → [cuda/stall.md](./cuda/stall.md)
- 用 profiler 打标记 → [cuda/nvtx.md](./cuda/nvtx.md)
- 统计模型参数量和 FLOPs → [thop/](./thop/README.md)
