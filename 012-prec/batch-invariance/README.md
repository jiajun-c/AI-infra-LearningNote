# batch-invariance

## 1. 定义

batch-invariance指的是kernel的输出可能由于batch的大小不同，或者说由于和不同的序列去拼接batch(continuous batching)而每次的输出不同

## 2. 原因

- 启发性算子的不确定性：例如gemm的batch size就是m维度，cuBlas按照(M, N, K)选择tile大小和split-K份数，不同的tile大小和split策略都会导致最终输出的精度不确定
- 规约类算子：有些规约类的算子是使用原子加实现的，会由于CTA调度顺序的不确定性导致精度不确定
- 通信操作：有些通信算法也是启发性的，规约顺序会变化，例如有时候选择ring规约，有时候选择tree规约

## 3. 解决方法

- persist kernel：限定每个sm上要执行的任务，每个输出tile由单个CTA沿着K顺序累加，天然batch-invariant
- 固定一些kernel的tile shape
- 采用固定的规约顺序
- 限定NCCL的拓扑算法
