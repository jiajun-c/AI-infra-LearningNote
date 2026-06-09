# Copy Engine

GPU 上独立于 SM（Streaming Multiprocessor）的 **DMA 硬件单元**，专门负责内存拷贝，不占用 CUDA Core 算力。

## 1. 什么年代有的

Copy Engine 不是 Blackwell 的新东西，历史很长：

| 架构 | 代 | 年份 | Copy Engine |
|------|----|------|------------|
| Tesla | G80 | 2006 | **无** — 拷贝和计算共用 SM |
| Fermi | GF100 | 2010 | **1 个** — 首次独立出 Copy Engine |
| Kepler | GK104 | 2012 | 1-2 个, 支持双向并发 |
| Maxwell | GM204 | 2014 | 2 个 (H2D + D2H 分离) |
| Pascal | GP102 | 2016 | 2 个, GPUDirect RDMA |
| Volta | GV100 | 2017 | 多路并发拷贝 |
| Ampere | GA100 | 2020 | 多引擎, 配合 L2 cache |
| Hopper | GH100 | 2022 | 硬件加速的异步拷贝 (TMA) |
| Blackwell | GB100 | 2024 | **更多引擎、更高带宽** |

**Blackwell 不是发明了 Copy Engine，而是改进了它。**

## 2. 硬件视角

```
                      GPU
  ┌─────────────────────────────────────────────┐
  │                                             │
  │   SM 0      SM 1      ...     SM N         │
  │   (计算)    (计算)           (计算)          │
  │     │         │                │            │
  │     └─────────┼────────────────┘            │
  │               │                              │
  │         Crossbar / L2                       │
  │               │                              │
  │     ┌─────────┼─────────┐                   │
  │     │         │         │                   │
  │  Copy Engine  │    Copy Engine              │
  │   (H2D)      │      (D2H)                  │
  │     │         │         │                   │
  └─────┼─────────┼─────────┼───────────────────┘
        │         │         │
    ────┼─────────┼─────────┼─── PCIe / NVLink ────
        │         │         │
        ▼         ▼         ▼
      CPU Memory (pinned)    Remote GPU
```

Copy Engine 和 SM 完全独立，共享同一个内存控制器（L2 → DRAM），所以可以**同时工作**：

```text
时间 →
SM:       [kernel A] [kernel B] [kernel C]
Copy Eng:   [H2D]              [D2H]
           ↑ 计算和数据搬运并发, 互不阻塞
```

## 3. 软件使用

Copy Engine 通过 **异步 CUDA API** 调用，不需要用户显式指定：

```cpp
// 异步拷贝 → 自动使用 Copy Engine
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(dst);  // 同一 stream = 顺序执行
cudaMemcpyAsync(host, dst, size, cudaMemcpyDeviceToHost, stream);

// 如果想重叠:
cudaMemcpyAsync(d_buf, h_data, size, cudaMemcpyHostToDevice, stream1);
kernel<<<grid, block, 0, stream2>>>(d_other);  // 不同 stream, 并发!
```

**关键规则**：

| 操作 | 使用的硬件 | 与 SM 并发？ |
|------|-----------|-------------|
| `cudaMemcpy` (同步) | Copy Engine + **阻塞 CPU** | 不能 (CPU 被占) |
| `cudaMemcpyAsync` (异步) | Copy Engine | **可以** |
| Kernel `<<<>>>` | SM | — |
| `cudaMemcpyAsync` (default stream) | Copy Engine | 不能 (隐式同步) |
| `cudaMemcpyAsync` (non-default stream) | Copy Engine | **可以** |

## 4. 每代 GPU 有多少 Copy Engine

可以用 `cudaGetDeviceProperties` 查看：

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Async Engine Count: %d\n", prop.asyncEngineCount);
```

典型值：

| GPU | asyncEngineCount | 说明 |
|-----|-----------------|------|
| GTX 1080 | 2 | H2D + D2H 各 1 |
| RTX 3080 | 2 | H2D + D2H 各 1 |
| RTX 4060 Laptop | 1 | 笔记本, H2D/D2H 共用 |
| A100 | 2+ | 多路并发 |
| H100 | 4+ | TMA 进一步加速 SM↔GMEM |

## 5. Hopper 的新玩法：TMA

Hopper (H100) 引入了 **TMA (Tensor Memory Accelerator)**，是 Copy Engine 的进化版：

```
传统 Copy Engine:  CPU ↔ GPU (跨 PCIe/NVLink)
TMA:               SM ↔ Global Memory (GPU 内部)
```

TMA 在硬件层面完成 `global→shared` 的异步拷贝，不需要 SM 的 Load/Store 指令参与。这是 Copy Engine 理念在 GPU 内部的延伸。

## 6. 常见误解

| 误解 | 事实 |
|------|------|
| "Copy Engine 是 Blackwell 新加的" | Fermi (2010) 就有了，Blackwell 只是增强了 |
| "cudaMemcpyAsync 自动重叠" | 必须用不同 stream，且源/目标内存要 pinned |
| "重叠就是免费的性能" | Copy Engine 带宽有限（~PCIe 带宽），多个拷贝之间也会竞争 |
| "流越多重叠越好" | Copy Engine 数量有限（通常 1-2 个），多余的流要排队 |
