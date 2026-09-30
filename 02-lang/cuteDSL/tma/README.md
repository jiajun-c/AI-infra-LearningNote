# cuteDSL tma

`cute.nvgpu.cpasync.create_tma_multicast_mask` 可以创建`mcast_mode`维度编号，但是其他维度和`cta_layout_vmnk`中的保持一致

## 1. tma_partition

对gmem进行划分，返回的有两个tensor，第一个是CTA的共享内存的tile，第二个是CTA的全局内存的tile，传入的smem_tensor和gmem_tensor的shape的第一个维度需要等于tile的shape

```python
tma_partition(
    atom: cute.CopyAtom,
    cta_coord,
    cta_layout: cute.Layout,
    smem_tensor: cute.Tensor,
    gmem_tensor: cute.Tensor
) -> tuple[cute.Tensor, cute.Tensor]
```

往往会通过group_modes的接口来把前面的几个维度组织成一个子维度从而对应tile的shape

```cpp
tma_s, tma_src = cute.nvgpu.cpasync.tma_partition(
    load_atom,
    0,
    cute.make_layout(1),
    cute.group_modes(s_tile, 0, 2),
    cute.group_modes(g_src_tiles, 0, 2)
)
```

然后发起tma拷贝，会更新barrier的到达字节数

```cpp
cute.copy(
    load_atom,
    tma_src[(None, block_m, block_n)],
    tma_s,
    tma_bar_ptr=barrier
)
```

## 2. tma的同步机制

tma中有两种同步机制

- mbarrier
- bulk group

### 2.1 mbarrier

mbarrier用于数据从Global到Shared，用预期到达的字节数量来表示是否完成，因为数据写入到shared 是通过mbarrier来报告的

### 2.2 bulk group

用于等待从shared 到 Global的操作，因为tma的store指令是通过bulk 

```python
    cute.copy(
        load_atom,
        tma_src[(None, block_m, block_n)],
        tma_s,
        tma_bar_ptr=barrier
    )

    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(barrier)
    cute.arch.mbarrier_wait(barrier, 0)

    cute.copy(store_atom, tma_s, tma_dst[(None, block_m, block_n)])

    cute.arch.cp_async_bulk_commit_group()
    cute.arch.cp_async_bulk_wait_group(0, read=False)
```

read为False表示这个指令会等待写出完成，而read=True的时候将会当已经从share memory读取完成后就返回，从而可以释放share memory的占用

## 3. 性能优化

每次需要一个tma描述符描述硬件这块张量如何存储，以及一次要搬哪种形状的tile，tma指令通过这个描述符来自行计算地址并搬运数据，不需要线程逐个元素计算地址

通过`prefetch_descriptor`来提前让硬件取得这份描述信息，而不是在tma进行搬运的时候才进行去把这个描述符加载进来

```python
with cute,arch.elect_one():
    cute.nvgpu.cpasync.prefetch_descriptor(load_atom)
    cute.nvgpu.cpasync.prefetch_descriptor(store_atom)
```

## 4. tma广播

tma广播在从HBM上load数据时从一个CTA给广播到处于一个cluster内的其他CTA的共享内存上。

使用`mcast_mask`来表示cast到cluster中的哪几个CTA中

```python
@cute.kernel
def multicast_kernel(atom: cute.CopyAtom, g_src: cute.Tensor, out: cute.Tensor):
    rank = cute.arch.block_idx_in_cluster()
    tile_idx = cute.arch.block_idx()[0]//CLUSTER_SIZE
    lane = cute.arch.thread_idx()[0]
    storage = SmemAllocator().allocate(SharedStorage)
    barrier = storage.barriers.data_ptr()
    s_tile = storage.tile.get_tensor(cute.make_layout(TILE))
    g_tile = cute.local_tile(g_src, (TILE, ), (None, ))

    cta_layout = cute.make_layout(1)
    tma_s, tma_g = cute.nvgpu.cpasync.tma_partition(
        atom, 0, cta_layout, s_tile, g_tile
    )

    with cute.arch.elect_one():
        cute.arch.mbarrier_init(barrier, 1)
        cute.arch.mbarrier_expect_tx(barrier, TILE*4)
    cute.arch.mbarrier_init_fence()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()

    if rank == 0:
        cute.copy(
            atom.with_(
                tma_bar_ptr=barrier,
                mcast_mask=cutlass.Int16((1<<CLUSTER_SIZE)-1),
            ),tma_g,tma_s,)
```