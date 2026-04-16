# NVIDIA H100 (Hopper 架构) 缓存规格

## H100 80GB HBM3 规格

| 参数 | 值 |
|------|-----|
| **架构** | Hopper |
| **Compute Capability** | 9.0 |
| **SM 数量** | 132 |
| **L2 Cache** | 50 MB |
| **L2 Cache per SM (平均)** | ~390 KB |
| **Shared Memory / L1 per SM** | 192 KB (可配置) |

## 内存层次结构

```
H100 (132 SMs)
├── L2 Cache (50 MB, 所有 SM 共享)
│   ├── 分为多个 partition (通常 16-20 个)
│   └── 每个 partition 有自己的内存控制器
│
└── Per-SM Cache
    ├── L1/Shared Memory: 192 KB
    │   ├── 可配置比例 (如 128KB Shared + 64KB L1)
    │   └── 用于 block 内线程共享数据
    ├── Texture Cache
    └── Constant Cache
```

## L2 Cache 分布

H100 的 50 MB L2 Cache 不是均匀的，而是分为多个 **L2 Partition**：
- 每个 L2 partition 服务于一组 SM
- SM 访问 L2 时，地址决定去哪个 partition
- **交错的 SM 放置**可能更好地分散到不同 L2 partition

---

# 性能结果分析

## 测试数据回顾

| Shape | Sequential (GB/s) | Interleaved (GB/s) | 差异 |
|-------|-------------------|---------------------|------|
| 512 x 1024 | 493.08 | 465.99 | **-5.49%** |
| 1024 x 2048 | 916.01 | 904.33 | **-1.27%** |
| 2048 x 4096 | 956.99 | 1141.52 | **+19.28%** |
| 4096 x 4096 | 600.47 | 597.97 | **-0.42%** |
| 4096 x 8192 | 615.56 | 615.51 | **-0.01%** |
| 8192 x 8192 | 630.51 | 622.04 | **-1.34%** |

---

## 分析

### 1. 为什么大多数情况下差异不大？

**GEMV 是内存带宽受限 (Memory-Bound) 操作**：
- 每个元素只读取一次，没有数据复用
- 性能主要取决于 HBM 带宽，而不是 L2 命中
- H100 的 HBM3 带宽 ~3.35 TB/s，是瓶颈所在

**L2 Cache 对 GEMV 帮助有限**：
- 矩阵 A：67M 元素 (2048x4096)，约 256 MB >> 50 MB L2
- 向量 X：被重复读取，但大小远小于 L2，两种策略都能受益
- 向量 Y：写操作，不经过 L2 (write-through)

### 2. 为什么 2048 x 4096 时 Interleaved 快 19%？

这个形状下：
- 矩阵 A = 2048 × 4096 × 4B = **32 MB**
- 向量 X = 4096 × 4B = **16 KB**
- 向量 Y = 2048 × 4B = **8 KB**

**32 MB 的矩阵刚好能部分放入 L2**！

**Interleaved 优势**：
- 偶数 SM (0, 2, 4, ...) 分布在不同的物理区域
- 访问不同的 L2 partition，减少冲突
- L2 partition 之间的带宽叠加

**Sequential 劣势**：
- 连续 SM (0-65) 可能访问相邻的 L2 partition
- 造成 L2 partition 热点，带宽受限

### 3. 为什么小矩阵 (512x1024) Interleaved 反而慢？

- 矩阵 A = 512 × 1024 × 4B = **2 MB** << 50 MB L2
- 整个矩阵都能放入 L2
- **Sequential SM 可能有更好的 L1/Shared Memory 利用**
- Interleaved 的 SM 分布导致 L1 预取效率降低

### 4. 为什么大矩阵 (8192x8192) 差异小？

- 矩阵 A = 8192 × 8192 × 4B = **256 MB** >> 50 MB L2
- L2 几乎 miss，直接从 HBM 读取
- 性能由 HBM 带宽决定，L2 分布影响小

---

## 结论

| 场景 | 哪种策略更好 | 原因 |
|------|-------------|------|
| **小矩阵 (<< L2)** | Sequential | L1/Shared 利用更好 |
| **中等矩阵 (~L2/2)** | Interleaved | L2 partition 分散，减少热点 |
| **大矩阵 (>> L2)** | 无明显差异 | HBM 带宽是瓶颈 |

**实际建议**：
- 对于 GEMV 类操作，SM 放置策略影响有限（<5%）
- 在中等矩阵大小时，Interleaved 可能有意外收益
- 更应关注：occupancy、内存合并访问、L1/Shared 配置
