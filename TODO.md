# 学习 TODO

基于现有目录梳理出的待学习内容，按优先级排列。

---

## 01 CUDA

- [ ] **Graph Capture**：`cudaGraphCapture` + `cudaGraphLaunch`，训练循环固化为 CUDA Graph 消除 kernel launch overhead
- [ ] **Multi-Stream 通信计算重叠**：stream 依赖关系、event 同步、与 NCCL 的配合
- [ ] **Persistent Kernel**：长驻 kernel + ring buffer，避免反复 launch，适合 decode 阶段
- [ ] **NVLink / NVSwitch 拓扑感知**：`cudaDeviceGetP2PAttribute`，peer access，带宽建模
- [ ] **Blackwell 新特性**：FP4 Tensor Core、第五代 NVLink、CTA cluster 扩展

---

## 02 编程语言

### C++
- [ ] **无锁数据结构**：`std::atomic`、CAS、lock-free queue，AI infra 中的生产者消费者场景
- [ ] **内存模型**：`memory_order`，acquire/release 语义，和 GPU `__sync_synchronize` 的对比
- [ ] **模板元编程**：CRTP、type traits，CUTLASS/CuTe 大量使用

### Python
- [ ] **GIL 与多线程**：GIL 的本质、何时释放、与 asyncio/multiprocessing 的边界
- [ ] **`__torch_dispatch__` / `__torch_function__`**：PyTorch 算子拦截机制

### Triton
- [ ] **Flash Attention 实现**：在 Triton 中实现 online softmax + tiling
- [ ] **Persistent kernel 模式**：grid-stride loop，减少 kernel launch
- [ ] **Triton 与 torch.compile 集成**：`@triton.jit` + `inductor` 后端

---

## 03 LLM

### 推理
- [ ] **PagedAttention**：vLLM 的 KV cache 分页管理，block table 实现
- [ ] **Continuous Batching**：iteration-level scheduling，与 static batching 的对比
- [ ] **Chunked Prefill**：prefill/decode 解耦调度，减少 decode 抖动
- [ ] **MLA（Multi-head Latent Attention）**：DeepSeek 的 KV cache 压缩方案
- [ ] **投机解码进阶**：SpecInfer、Medusa、Eagle 多草稿头

### 训练
- [ ] **序列并行（SP）**：Megatron-SP，ring attention，长序列训练
- [ ] **ZeRO-3 + Offload**：参数/梯度/优化器状态的 CPU offload 细节
- [ ] **异步 checkpoint**：训练不停顿的 checkpoint 写入（torch.distributed.checkpoint）
- [ ] **MoE 训练**：Expert 并行、负载均衡 loss、token dropping

### 模型结构
- [ ] **RoPE 变体**：YaRN、LongRoPE，长上下文位置编码
- [ ] **GQA / MQA**：grouped/multi-query attention 的 KV 共享实现

---

## 04 通信

- [ ] **All-to-All**：MoE expert dispatch 的核心通信原语，与 All-Reduce 的对比
- [ ] **通信计算重叠**：`isend/irecv` + `wait`，Tensor Parallel 中的 overlap 技术
- [ ] **RDMA / RoCE**：AI 集群网络基础，`ibverbs`，与 NCCL 的关系
- [ ] **集合通信拓扑算法**：Ring、Tree、Recursive Halving，带宽利用率分析

---

## 05 框架

### PyTorch
- [ ] **torch.compile 深入**：`dynamo` 图捕获、`inductor` 代码生成、graph break 诊断
- [ ] **自定义 autograd Function**：`forward/backward`、`ctx.save_for_backward`、梯度检查
- [ ] **`torch.export`**：静态图导出，与 `torch.jit.script` 的区别，部署场景

### vLLM
- [ ] **Scheduler 源码**：`waiting/running/swapped` 三队列状态机
- [ ] **Worker / Engine 架构**：`AsyncLLMEngine`，多进程 Worker，KV cache 初始化流程
- [ ] **prefix caching**：RadixAttention，prompt 共享前缀的 KV 复用

### Megatron-LM
- [ ] **Interleaved Pipeline**：1F1B vs interleaved，bubble rate 计算
- [ ] **Distributed Optimizer**：梯度分片 + 参数分片的实现细节

---

## 06 Agent

- [ ] **RAG 进阶**：Hybrid Search（BM25 + 向量），reranker，chunk 策略
- [ ] **Tool Use / Function Calling**：JSON schema 解析，并发工具调用
- [ ] **Memory 机制**：短期/长期记忆，MemGPT 的虚拟上下文管理

---

## 07 系统

- [ ] **io_uring 进阶**：Fixed buffer、SQPOLL 模式、链式 SQE（`IOSQE_IO_LINK`）
- [ ] **NUMA 感知内存分配**：`libnuma`，`numactl`，跨 NUMA 带宽惩罚
- [ ] **巨页（HugePage）管理**：`hugetlbfs`，预分配 2MB/1GB 页，和 THP 的区别
- [ ] **Linux 调度器**：CFS、实时调度类（`SCHED_FIFO`），AI 训练进程绑核

---

## 08 工具链

- [ ] **TVM / Relax**：算子融合、布局变换、auto-tuning
- [ ] **ONNXRuntime**：执行提供者（CUDA EP），优化 pass，与 PyTorch 互转
- [ ] **Triton Inference Server**：动态 batching、model ensemble、gRPC 接口

---

## 09 性能分析

- [ ] **Nsight Compute 深入**：roofline model、memory chart、warp stall 分析
- [ ] **Nsight Systems**：CPU-GPU 时序对齐，NCCL 通信可视化
- [ ] **PyTorch Profiler**：`torch.profiler.profile`，chrome trace，内存时序分析
- [ ] **性能建模**：Arithmetic Intensity 计算，带宽/算力瓶颈判断方法论

---

## 近期会话涉及的延伸话题

- [ ] **instant_tensor 源码**：分布式 safetensors 加载，多文件并发 AIO，pinned memory 直接 DMA
- [ ] **vLLM CPU offload 实现**：KV cache swap 的 AIO 写入路径
- [ ] **asyncio 与 uvloop**：基于 libuv 的事件循环替换，吞吐对比
