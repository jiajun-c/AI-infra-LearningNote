# Gloo

Gloo是一个集合通信库，其可以用于CPU，GPU和CPU-GPU的环境，同时其支持TCP，UDP，RDMA等多种底层传输协议，能根据网络环境自动选择最优方案。

在纯CPU集群或者CPU-GPU混合的集群下推荐使用

在torch中可以使用`torch.distributed.init_process_group`来直接选择gloo作为后端。
