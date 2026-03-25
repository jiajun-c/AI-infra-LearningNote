# Triton

## 1. 原理

Triton本质上是一种MLIR，其层级为

> triton-lang -> triton ir -> triton gpu dialect ->llvmir -> ptx -> SASS

在GPU上层级分为线程层级，warp层级，block层级，grid层级

Triton所面向的block层级，通过program_id去对应到特定的block，内部warp层级则是由编译器进行一个自动映射

## 2. Kernel Fusion

[Fusion 详细文档](./fusion/permuteFusion/README.md)

通过 Triton 实现算子融合（Kernel Fusion），减少显存搬运开销。以 permute fusion 为例：

**场景**：对 `[M, K, N]` 的 tensor 沿 K 维度求和，得到 `[M, N]`

**四种实现方案对比**：

| 方案 | 步骤 | 是否需要 contiguous | 是否支持任意 K |
|------|------|:-------------------:|:--------------:|
| Unfused | permute + contiguous + kernel | ✅ | ❌ (K 为 constexpr) |
| Tile | permute + contiguous + tile_kernel | ✅ | ✅ (分块循环) |
| Fused | 仅 fused_kernel (kernel 内跨步读取) | ❌ | ❌ (K 为 constexpr) |
| FusedTile | 仅 fused_tile_kernel (跨步读取+分块) | ❌ | ✅ (分块循环) |

**核心思想**：通过在 kernel 内部计算 stride 地址来代替 `permute + contiguous` 的显存数据搬运，将 3 步操作融合为 1 步。

