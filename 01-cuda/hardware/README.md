# 硬件架构

GPU的层级可以分为

GPC -> TPC -> SM 

GPC (Graphics Processing Cluster，图形处理集群)：
- 它是 GPU 内部最大的功能单元块。
- Hopper 架构引入了“线程块集群”（Thread Block Cluster）概念，该级别在硬件上对应于 GPC。
- GPC 内部包含一个专用的 SM-to-SM 网络，允许该 GPC 内的 SM 之间进行低延迟通信。

TPC (Texture Processing Cluster，纹理处理集群)：
- TPC 嵌套在 GPC 内部。在 Hopper 架构中，1 个 TPC 包含 2 个 SM。
- TPC 是硬件屏蔽（Floorsweeping）和掩码控制的物理单位。正如你提到的，Hopper 的底层掩码索引对应的是物理 TPC 单元，包括那些被禁用的单元。


SM (Streaming Multiprocessor，流式多处理器)：
- 它是 GPU 执行计算任务的基本单元。
- 每个 SM 拥有自己的寄存器堆（Register File）、L1 缓存/共享内存（在 Hopper 上为 256KB）、张量核心（Tensor Cores）以及执行单元。