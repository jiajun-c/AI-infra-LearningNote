# FlashAttention

## FlashAttention V1

- FLOPS：等同于FLOP/s，表示每秒执行的FLOPs数量
- FLOPs：表示某个算法的总计算量


$\pi$：算力硬件的上限，以H100为例，其FP16算力，1,979TFlops

$\beta$: 硬件带宽上限，以H100为例，其为 3.2T，1T = 1e12

所以其拐点为 1979 / 3.2  = 618FLOPs/Byte

当算数强度 < 618的时候，表示是Memory Bound， > 618的时候是 Compute bound

对于一个矩阵运算，其FLops可以计算为 2\*M\*N\*K，访存的量为 (M\*K + N\*K + M\*N)*2

其算数强度为 

$$I = \frac{2MNK}{2(MK + KN + MN)} = \frac{MNK}{MK + KN + MN} \text{ FLOPs/Byte}$$

大模型输入为 [S, D]，$Q \times K^T$ 的算数强度为 $\frac{S^2 D}{2SD + S^2}$

对于pefill 阶段，当seq_len比较长的时候，这部分会变为 compute bound

对于softmax操作而言，其算术强度为 

$I = \frac{5N}{4N \times 2 \text{ Bytes}} = \frac{5}{8} = \textbf{0.625 FLOPs/Byte}$ 

这是一个显然的memory bound

为了解决这样一个memory bound的问题，提出了flashAttention，把softmax的操作fusion进入了矩阵乘的计算中

为了解决这个问题，提出了flashAttention

flashAttention的出发点就是在Q * K^T算出小块的时候立刻和V相乘，然后利用online softmax的方式补偿分块max，所以K和V需要一起遍历

在v1的时候选择的是在最外层遍历K和V，在内层遍历Q，这样缺点就是在内层不断去访问Global Memory

