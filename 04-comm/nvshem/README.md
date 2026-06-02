# nvshem

nvshem和nccl的区别在于nvshem他是一个单向的通信，直接从本地GPU去访问其他GPU的数据，而nccl需要进行双向的通信，一个节点send，一个节点recv，其优势在于处理小批量的数据，对于多节点而言，nccl可以更好地利用网络的拓扑

总结一下优化
- 单边通信
- GPU原生通信
- offload到硬件通信，不抢占sm

