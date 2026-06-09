# GPU L1 / L2 Cache

## 1. 缓存层次结构

```text
线程私有               Block 共享              整个 GPU 共享
────────              ──────────              ────────────
寄存器                共享内存 (SMEM)          L2 Cache
(最快, ~0 cycle)      (~20-30 cycles)         (~200-300 cycles)
                      L1 Cache                 HBM / GDDR
                      (~30-50 cycles)          (~400-800 cycles)
                      
                      └── 同一片上 SRAM ──┘
                      L1 和 SMEM 共享物理存储
```

## 2. L1 Cache：可配置，不可直接编程

L1 和共享内存在 **同一块片上 SRAM** 上，物理上是同一块资源：

```text
SM 内部一块 128KB (或 256KB) SRAM
         │
         ├── L1 Cache    ← 硬件自动缓存, 程序员不直接控制
         │
         └── Shared Mem  ← 程序员显式管理 (__shared__)
```

### 2.1 分配比例可配置

```cpp
// 方式 1: 全局设置 L1 / SMEM 比例
cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);  // SMEM 优先 (100KB SMEM / 28KB L1)
cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);      // L1 优先 (28KB SMEM / 100KB L1)
cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);   // 均分

// 方式 2: 针对单个 kernel 设置
cudaFuncSetCacheConfig(myKernel, cudaFuncCachePreferShared);
```

| 配置 | L1 | SMEM | 适用场景 |
| ---- | -- | ---- | -------- |
| `PreferShared` | 小 | 大 | GEMM tiling (需要大 tile) |
| `PreferL1` | 大 | 小 | 稀疏访存 (依赖缓存命中) |
| `PreferEqual` | 均分 | 均分 | 通用 |

**这是"编程"L1 的唯一方式——控制大小，不控制内容。** L1 仍然是硬件管理的 cache：自动缓存全局内存访问，自动替换策略（LRU 之类），你没法显式 load/store 到 L1。

### 2.2 可以用 LDG 的缓存提示

从 SM 8.0 (Ampere) 开始，可以在 PTX 层面控制缓存行为：

```cpp
// PTX 层面: ld.global.ca  → 缓存到 L1 (cache at all levels)
//           ld.global.cg  → 缓存到 L2, 跳过 L1 (cache global)
//           ld.global.cs  → 流式, 尽量不缓存 (cache streaming)
//           ld.global.cv  → 标记为 volatile (不缓存)

// C++ 层面: __ldg() 强制走 L1 (只读缓存)
float x = __ldg(&global_array[idx]);  // 只读, L1 缓存

// 或使用 __ldcs() 跳过 L1, 只缓存到 L2
float y = __ldcs(&global_array[idx]);
```

## 3. L2 Cache：几乎不可编程，但可以留置

L2 是统一缓存，所有 SM 共享。从 SM 8.0 (Ampere) 开始支持 **持久化留置**：

```cpp
// 设置 L2 持久化留置大小 (Ampere+)
int max_persisting_bytes = 0;
cudaDeviceGetAttribute(&max_persisting_bytes,
    cudaDevAttrMaxPersistingL2CacheSize, deviceId);

// 设置当前留置窗口
size_t window_size = max_persisting_bytes / 2;  // 留置一半 L2
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, window_size);

// 在 kernel 中标记某些访问为"留置"
// 通过 stream 属性:
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr  = ptr;
attr.accessPolicyWindow.num_bytes = size;
attr.accessPolicyWindow.hitRatio  = 1.0;  // 希望 100% 留在 L2
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

效果：被标记的地址范围不会被 L2 的正常替换策略踢出，相当于在 L2 中留置一块"软 shared memory"。Hopper (H100) 把这项能力产品化了（TMA 配合 L2 留置做异步拷贝）。

## 4. 编程能力总结

| | 共享内存 (SMEM) | L1 Cache | L2 Cache |
| --- | --- | --- | --- |
| **物理介质** | 片上 SRAM | 片上 SRAM (与 SMEM 共享) | 独立 SRAM |
| **容量** | ~100KB/SM (可配) | ~28-100KB/SM (可配) | 40-60MB (固定) |
| **显式读写** | ✅ `__shared__` + load/store | ❌ 硬件自动 | ❌ 硬件自动 |
| **大小配置** | ✅ 与 L1 此消彼长 | ✅ 与 SMEM 此消彼长 | ❌ 固定 |
| **缓存策略** | N/A (显式管理) | 部分可控 (ld.global.ca/cg) | 留置窗口 (Ampere+) |
| **替换策略** | N/A (显式管理) | ❌ 硬件决定 | 部分可控 (留置标记) |
| **延迟** | ~20-30 cycles | ~30-50 cycles | ~200-300 cycles |

## 5. 实践建议

**普通 kernel 优化**：不需要碰 L1/L2 配置。用 shared memory 做显式的数据复用就够了。

**GEMM / FlashAttention 等重度 tiling**：`cudaFuncCachePreferShared` 给 SMEM 让路。CUTLASS/CuTe 内部自动处理这些。

**需要留置 L2 的场景**：少数情况。比如推理中 weight 矩阵反复被读、KV cache 被反复访问。在 Triton 里可以用 `tl.inline_asm_elementwise` 调 PTX 的 cache hint。

**核心思想**：GPU cache 设计哲学是"尽量让程序员不用管"，真正的性能杠杆在 shared memory 的显式管理。L1/L2 的可编程性是锦上添花，不是必需品。
