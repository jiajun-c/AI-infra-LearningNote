# 每一代Tensor core的变迁

## 1. SM70 (Volta)

指令族是mma，协作粒度为单个warp，主要指令的shape为`m8n8k4`，wmma的shape为`m16n16k16`

这个时候还是同步的计算，要先把数据先从global搬运到shared，再由warp load到寄存器，这个时候没有cp.async,都是同步的搬运和计算

## 2. SM80 (Ampere)

mma主要是新增了若干的类型的支持，同时加入了cp.async，可以做一个异步的搬运

`kernel`可以构建经典的流水

- stage0: 加载下一块的A/B
- stage1: 当前块的 mma.sync
- stage2: 写回了上一块的结构

## 3. SM90 (Hopper)

sm90上增加了wgmma，对于hopper上的普通mma指令，其实还是需要数据预先加载到线程的寄存器上。

对于`wgmma`来说，其可以讲操作数的位置放在共享内存上。

有两种模式

- SS：把A和B都放在共享内存上
- RS：把A放在寄存器上，把B放在共享内存上

同时sm90上加入了tma指令，可以进行异步的load，然后进行mma的计算

## 4. SM100(blackwell)

sm100上新增了tcgen05的指令，其可以一次处理128xNxK或者2-CTA模式下的256xNxK的指令，同时要注意的是Ampere和Hopper上的mma指令其实在sm100上也是兼容的。

而tcgen05的区别在于输出的元素位置其实可以放到tmem里面。同时2-CTA支持更大的M

