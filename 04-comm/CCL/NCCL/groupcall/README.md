# NCCL Group Call

`ncclGroupStart()` / `ncclGroupEnd()` 是 NCCL 中一个重要的机制。将多个 NCCL 操作打包成一个"组"，让 NCCL 内部可以对它们进行合并执行。

## 1. 基本用法

```cpp
ncclGroupStart();
// ... NCCL 操作 ...
ncclGroupEnd();
```

夹在 `GroupStart` 和 `GroupEnd` 之间的所有 NCCL 调用不会立即提交执行，而是被暂存起来，在 `GroupEnd` 时统一处理。

## 2. 核心用途

### 2.1 多 GPU 并行初始化

最常见的用法。`ncclCommInitRank` 需要所有 rank 同时参与，如果不加 group，每个 rank 的 init 调用会串行阻塞等待其他 rank，造成死锁或极慢的初始化。

```cpp
// ❌ 错误：每个 InitRank 会阻塞等待其他 rank
for (int i = 0; i < nGPUs; i++) {
    cudaSetDevice(devlist[i]);
    ncclCommInitRank(comms + i, nGPUs, id, i);  // 死锁！
}

// ✅ 正确：用 Group 包裹，所有 rank 同时一起初始化
ncclUniqueId id;
ncclGetUniqueId(&id);

ncclGroupStart();
for (int i = 0; i < nGPUs; i++) {
    cudaSetDevice(devlist[i]);
    ncclCommInitRank(comms + i, nGPUs, id, i);
}
ncclGroupEnd();  // 所有 rank 在这里一起完成初始化
```

同样的模式也适用于 `ncclMemAlloc`、`ncclCommWindowRegister` 等需要跨 GPU 协调的操作。

### 2.2 多 GPU 集合通信并行提交

管理多个 GPU 时，每个 GPU 上的通信操作需要同时发出。Group 确保所有 GPU 的操作被一起提交到各自的 stream：

```cpp
// 同时对所有 GPU 发起 AllReduce
ncclGroupStart();
for (int i = 0; i < nGPUs; i++) {
    ncclAllReduce(sendbuff[i], recvbuff[i], count,
                  ncclFloat, ncclSum, comms[i], streams[i]);
}
ncclGroupEnd();  // 所有 GPU 的 AllReduce 同时开始
```

## 3. 内部机制

不使用 Group 时，不同 GPU 上的 kernel launch 有先后，先到的 GPU 需要等待后到的 GPU：

```text
不使用 Group:
  GPU0: AllReduce launch → 等待 GPU1 到位 → 执行
  GPU1: AllReduce launch → 等待 GPU0 到位 → 执行
  问题：launch 有先后，造成"先到先等"

使用 Group:
  ncclGroupStart()         — 开始暂存
  GPU0: AllReduce → 暂存，不下发
  GPU1: AllReduce → 暂存，不下发
  ncclGroupEnd()           — 一起下发到所有 GPU
  结果：所有 GPU 的 kernel 几乎同时到位，减少等待时间
```

GroupEnd 内部会进行 merge/fusion 优化：

- 将同一个 group 内的多个操作合并为一个 NCCL kernel 下发
- 减少 CUDA kernel launch 的 overhead
- 避免 GPU 间的"先到先等"问题

## 4. 集体操作融合 (Collective Fusion)

NCCL 在 Group 内部可以将多个独立的集合通信操作融合为一个 kernel：

```cpp
ncclGroupStart();
ncclAllReduce(src1, dst1, n, ncclFloat, ncclSum, comm, stream);  // 操作 A
ncclAllReduce(src2, dst2, n, ncclFloat, ncclSum, comm, stream);  // 操作 B
ncclGroupEnd();
// NCCL 可能将 A 和 B 合并为一个 kernel，一次完成两个 AllReduce
```

这在 PyTorch 的 `torch.distributed` 中也通过 `_batch_p2p` 等接口间接使用。

## 5. 与 ncclCommInitAll 的关系

```cpp
// ncclCommInitAll 本质上就是 Group + InitRank 的封装
ncclCommInitAll(comms, nGPUs, devlist);

// 等价于：
ncclUniqueId id;
ncclGetUniqueId(&id);
ncclGroupStart();
for (int i = 0; i < nGPUs; i++) {
    cudaSetDevice(devlist[i]);
    ncclCommInitRank(comms + i, nGPUs, id, i);
}
ncclGroupEnd();
```

但当需要传入 `ncclConfig_t` 自定义参数时（如设置 CTA 数量、启用 Zero-CTA），就必须手动使用 Group + `ncclCommInitRankConfig`：

```cpp
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
config.maxCTAs = 4;  // 限制 CTA 数量

ncclGroupStart();
for (int i = 0; i < nGPUs; i++) {
    cudaSetDevice(devlist[i]);
    ncclCommInitRankConfig(comms + i, nGPUs, id, i, &config);
}
ncclGroupEnd();
```

## 6. 常见使用模式总结

| 场景 | 模式 |
| --- | --- |
| 多 GPU 初始化 | `GroupStart` → `InitRank` × N → `GroupEnd` |
| 带配置的多 GPU 初始化 | `GroupStart` → `InitRankConfig` × N → `GroupEnd` |
| 多 GPU 通信 | `GroupStart` → 通信操作 × N → `GroupEnd` |
| 通信+内存操作 | `GroupStart` → `MemAlloc` × N + `WindowRegister` × N → `GroupEnd` |
| 单 GPU 多操作融合 | `GroupStart` → 多个通信 op → `GroupEnd`（NCCL 可能 fuse） |

## 7. 注意事项

- **不要嵌套 Group**：`GroupStart` 内部不能再调用 `GroupStart`
- **不能跨 Group 的异步操作依赖**：如果操作 A 在 Group1，操作 B 在 Group2，B 依赖 A 的结果，需要 group 之间的 stream 同步
- **Group 内操作不能太多**：过多的操作可能导致 NCCL 内部 buffer 溢出或融合后的 kernel 过大
