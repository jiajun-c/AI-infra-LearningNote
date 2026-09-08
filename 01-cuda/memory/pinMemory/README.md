# GPU pin memory

pin memory是锁页内存，用于pytorch训练数据加载，H2D，D2D的拷贝，他更快的原因有两个

- 数据传输本身少一次拷贝，没有pageable->pinned临时缓冲的那一次
- 能于GPU计算去做overlap，而pageable是需要同步等待

## 1. 原因1

pageable memory必须经过pinned staging buffer，但是pageable memory的物理页可能随时被os换出，所以GPU不能直接对pageable内存发起dma

## 2. 原因2

pinned 的物理地址稳定，driver 可以直接对源 buffer 发起 DMA：


[pinned host buffer]
        ↓ DMA (pinned → device)        ← 一次 DMA
[GPU global memory]