# 通信与网络

这个目录整理 AI infra 中的通信栈：从分布式训练里的通信模式，到 NCCL/Gloo 等通信库，再到 NVLink、RDMA/RoCE、数据中心网络和性能排障。

建议按下面的顺序学习：

```text
通信模式 -> 集合通信算法 -> 单机 GPU 互联 -> 跨机网络 -> NCCL/Gloo 后端 -> overlap 与排障
```

## 学习入口

| 主题 | 说明 |
| --- | --- |
| [集合通信原语](./collective/README.md) | AllReduce、AllGather、ReduceScatter、All-to-All 的语义和基本代码 |
| [NCCL](./CCL/NCCL/README.md) | GPU 集合通信、P2P、拓扑感知、性能优化入口 |
| [NCCL 配置](./CCL/NCCL/config/README.md) | CTA 数量、zero-CTA、对称内存、buffer 注册等调优点 |
| [NCCL Buffer](./CCL/NCCL/buffer/README.md) | 注册通信 buffer，减少通信对 SM 的占用 |
| [NCCL CUDA Graph](./CCL/NCCL/graph/README.md) | 把通信放入 CUDA Graph 时的一致性要求 |
| [NCCL MultiMem](./CCL/NCCL/multiMem/README.md) | NVSwitch 层级上的规约能力 |
| [Gloo](./backend/gloo/README.md) | CPU 或 CPU/GPU 混合集群通信后端 |
| [NVLink / NVSwitch](./nvlink/README.md) | 单机多 GPU 的高速互联硬件 |
| [NVSHMEM](./nvshem/README.md) | 单边通信、GPU 原生通信和小粒度远端访问 |
| [计算通信 overlap](./overlap/README.md) | 通信 stream、计算 stream、SM 预留与 overlap 实验 |

## 和训练并行的对应关系

| 并行策略 | 典型通信 | 网络关注点 |
| --- | --- | --- |
| DDP | AllReduce | 梯度 bucket、ring/tree 算法、跨机带宽 |
| FSDP / ZeRO | AllGather + ReduceScatter | 参数分片粒度、prefetch、通信隐藏 |
| TP | AllReduce / AllGather / ReduceScatter | 层内高频通信，适合放在 NVLink/NVSwitch 域内 |
| PP | Send / Recv | micro-batch 调度、stage 间延迟、气泡 |
| EP / MoE | All-to-All | token dispatch/combine，最依赖跨机网络和拥塞控制 |
| 长序列并行 | Ring / P2P / AllGather | attention block 的通信顺序与显存压力 |

## 网络方向建议补充

当前仓库已经有通信库和单机互联的基础，但跨机网络层还比较薄。后续建议按优先级补齐：

1. **RDMA / RoCE 基础**：QP、CQ、MR、doorbell、send/recv、read/write、atomic；理解为什么 GPUDirect RDMA 可以绕过 CPU 拷贝。
2. **InfiniBand 与以太网 RoCE 对比**：可靠传输、PFC/ECN、拥塞控制、lossless fabric、交换机 buffer。
3. **NCCL 网络路径**：`NCCL_SOCKET_IFNAME`、`NCCL_IB_HCA`、`NCCL_IB_GID_INDEX`、`NCCL_NET_GDR_LEVEL`、NCCL net plugin、PXN、CollNet。
4. **拓扑建模**：GPU-NIC 亲和性、PCIe switch、NUMA、rail-optimized 拓扑、multi-rail 带宽聚合。
5. **集合通信算法建模**：ring、tree、recursive doubling/halving、hierarchical allreduce；区分 latency-bound 和 bandwidth-bound。
6. **MoE All-to-All 专题**：token permutation、dispatch/combine、expert parallel、跨机 A2A 的尾延迟和负载不均衡。
7. **排障工具链**：`ibstat`、`ibv_devinfo`、`ib_write_bw`、`perftest`、`ethtool`、`nvidia-smi topo -m`、NCCL debug log、Nsight Systems。
8. **生产网络指标**：链路带宽、P99 latency、重传、ECN mark、PFC pause、端口丢包、NCCL bus bandwidth、训练 MFU 下降定位。

## 实践路径

```text
1. 单机：nvidia-smi topo -m -> p2p bandwidth -> NCCL all_reduce_perf
2. 双机：ib_write_bw -> NCCL socket/IB 对比 -> GPUDirect RDMA 验证
3. 多机：all_reduce_perf / all_gather_perf / alltoall_perf -> 拓扑与 rank 映射
4. 训练：DDP/FSDP/TP/EP trace -> 找通信热点 -> 做 bucket、overlap、rank placement 调整
```

## 常见判断

- 如果 TP 通信成为瓶颈，优先检查是否跨了慢速互联，TP group 尽量放在 NVLink/NVSwitch 域内。
- 如果 DDP/FSDP 跨机性能差，先用 NCCL tests 和 RDMA perftest 把网络基线打出来，再看框架层 bucket 和 overlap。
- 如果 MoE 训练抖动明显，重点看 All-to-All 的 token 分布、rank 映射、跨 rack 流量和尾延迟。
- 如果通信与计算 overlap 不理想，检查 NCCL CTA 占用、buffer 注册、stream 顺序、CUDA event 依赖和 kernel 是否抢占同一批 SM。
