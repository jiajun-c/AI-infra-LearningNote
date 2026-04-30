# FSDP

## 1. 基础概念

FSDP(Fully Sharded Data Parallel)

每个GPU上保存一部分的模型参数，梯度，优化器状态的切片，每个GPU只持有其中的一部分

当需要计算到那一层的时候，进行AllGather的通信，然后完成后立刻释放，其相比于TP并行的优点，在于其延迟没有那么敏感，可以每次提前加载下一层的权重，适合进行机间的scale，而TP适合机内

