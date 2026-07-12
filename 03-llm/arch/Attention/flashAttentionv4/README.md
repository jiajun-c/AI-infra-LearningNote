# flashAttention v4

## 1. 介绍

flashAttentionv4是为了blackwell架构设计的attention，使用cute-DSL编写

## 2. 并行模式

整体的并行方式是外层Q，然后内层KV，但是fa4在工程实现上

- 0-3: softmax0_warp_ids：第0级softmax
- 4-7: softmax1_warp_ids：第1级softmax
- 8-11: correction_warp_ids，对 O 做 rescale 修正（rescale_O）、写回 tmem
- 12: mma，唯一的MMA的warp
- 13: epilogue_warp_ids，把O从smem到gmem
- 14: 发 TMA load 搬 Q/K/V（非 TMA 时变 2 warp）
- 15: empty warp id，充当clc调度器

## 3. 调度器

fa4和之前的架构有个不同的地方就是设计了调度器接口从而确定每个CTA分配哪个tile去进行计算

核心抽象是TileSchedulerProtocol，所有调度器都要实现

- initial_work_tile_info() 获取初始的tile
- advance_to_next_work() 推进到下一个tile
- prefetch_next_work() 提前预取下一个tile
- producer_tail() 消费完所有的tile

### 3.1 SingleTileScheduler

最简单的一对一调度

- blockIdx.x -> m_block
- blockIdx.y -> head
- blockIdx.z -> batch

### 3.2 StaticPersistentTileScheduler

持久化kernel的基础调度器，CTA 数量少于总 tile 数，每个 CTA 处理完一个 tile 后，跳 gridDim.x 步处理下一个，直到处理完所有 tile。

线性 tile 索引 → (block, head, batch) 的映射：

```cpp
hn_idx, block = divmod(tile_idx, num_block_cluster)
batch, head   = divmod(hn_idx, num_head)
```

### 3.3 SingleTileLPTScheduler

L2 Swizzle（缓存优化）：将 head 维度按 L2 容量分组（swizzle 参数），同一组内的 head 共享 K/V 数据。swizzle 的计算逻辑：

```cpp
size_one_head = seqlen_k * (headdim + headdim_v) * element_size
swizzle = L2_size // size_one_head  （向下取整到 2 的幂次）
```

此时tile的顺序是先在L2容量内切换head，再去切换tile

### 3.4 SingleTileVarlenScheduler

处理变长序列

- Warp 级前缀和：32 个 lane 并行计算各自 batch 的 num_m_blocks
- warp prefix sum 确定当前 tile_idx 落在哪个 batch 的哪个 block
- 区间搜索：以 31 个 batch 为一组，向前跳跃搜索
- 支持 LPT 和 head swizzle 的变长版本
