# NCCL cudagraph

## 1. 为什么不支持cuda-graph

在nccl2.9之前的nccl中其实都是不支持使用cuda-graph进行捕获的，因为在调用集合通信的时候，CPU会先选择任务并写入NCCL的channel work FIFO，setuplaunch()再查看FIFO，确定要启动多少block，哪个kernel，以及第一个kernel参数

同时本次网络任务需要被交给proxy线程，而proxy线程上的任务不会被回放

## 2. 如何支持cuda-graph

NCCL中的2.9 开始支持了cuda图接口，但是需要保证捕获时的一致性，当rank0在录制时，其他rank的节点也需要处于录制状态。

单节点上的流程可以分为下面的四步

- 识别捕获状态：判断stream是否在进行捕获，并取得当前graph
- 捕获时保存通信任务：NCCL计算算法，channel，kernel参数和proxy参数，存入ncclQueueInfo，随后通过cudaGraphAddHostNode加入一个CPU节点
- 捕获GPU kernel
- 每次回放时重新准备通信
