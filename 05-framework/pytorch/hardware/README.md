# SM 相关

## 1. `_SMCarveout_EXPERIMENTAL`

现在有些GPU现在的matmul使用的是persist kernel，相比于普通的kernel其没有sm的切换

```shell
传统 kernel：          Persistent kernel：
┌─────────────────┐    ┌───────────────-──┐
│ Block 1         │    │ Block 1 → Tile1  │
│ Block 2         │    │          ↓ Tile2 │
│ Block 3         │    │          ↓ Tile3 │
│ ...             │    │          ↓ ...   │
│ Block N         │    │ Block N → Tile1  │
└─────────────────┘    │          ↓ Tile2 │
每个 Block 处理         └────────────────-─┘
一个输出 tile         每个 Block 处理多个 tile
                        (软件流水线隐藏延迟)
```

关键点
- Persistent kernel 为每个 SM 启动 一个 CUDA block
- 每个 block 循环处理多个 output tile
- 通过软件流水线（software pipelining）隐藏 epilogue 等开销
- 适合每个 SM 只能容纳一个 tile 的情况

假设对于这样的persist kernel，同时启动一个NCCL，那么就会发生SM的抢占，从而使得kernel的执行时间出现增加

所以torch中提出了 `_SMCarveout_EXPERIMENTAL` 去为NCCL或者别的操作去预留SM