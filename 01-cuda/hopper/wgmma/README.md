# WGMMA（Warp Group MMA）

Hopper 架构引入的 Warp Group 级矩阵乘法指令。以 **4 个 Warp（128 线程）** 组成 Warp Group 执行 MMA，相比 Ampere 的 warp 级 MMA 有数量级吞吐提升。

## 核心特点

- **指令规模**：单条 WGMMA 处理 64×64×16 tile
- **直读 SMEM**：操作数直接来自 Shared Memory，无需先加载到寄存器
- **异步执行**：`warpgroup_arrive()` → `gemm()` → `warpgroup_commit_batch()` → `warpgroup_wait<0>()`
- **K-major 布局**：要求 SMEM 中矩阵按 K 维连续，需配合 swizzle

## 代码

| 文件 | 内容 |
|------|------|
| [demo.cu](demo.cu) | WGMMA + TMA + Pipeline 完整 GEMM 实现 |

## 架构文档

详见 [Hopper 架构专题](../../hardware/hopper.md#3-wgmmawarp-group-mma)
