# PyTorch Pin 内存实现详解

## 1. 整体架构

- `pin_memory` 实现

```shell
torch.Tensor.pin_memory()
    ↓
aten/src/ATen/native/Memory.cpp  (C++ 入口)
    ↓
globalContext().getPinnedMemoryAllocator()
    ↓
CUDAHooks → CUDACachingHostAllocator()
    ↓
cudaMallocHost() 或 cudaHostRegister()
```

## 2. Python API层

```shell
Tensor pin_memory(const Tensor& self, std::optional<c10::Device> device) {
  if (device.has_value()) {
    TORCH_WARN_DEPRECATION(
        "The argument 'device' of Tensor.pin_memory() ",
        "is deprecated. Please do not pass this argument.")
  }
  // Kind of mad that I have to do two dynamic dispatches here, pretty
  // annoying
  if (self.is_pinned(device)) {
    return self;
  }
  return at::_pin_memory(self, device);
}
```

- 从 globalContext() 获取 pinned 内存分配器
- 创建新的 CPU tensor，使用 pinned 存储
- 将原 tensor 数据复制过去

## 3. C++ 分配器层：CachingHostAllocator

这些最后会到C++的CachingHostAllocatorImpl

```cpp
struct CachingHostAllocatorImpl {

  using BlockPool = HostBlockPool<S, E, B>;
  using PrivatePool = HostPrivatePool<S, E, B>;

  virtual ~CachingHostAllocatorImpl() {
    if (active_) {
      active_ = false;
      getBackgroundThreadPool()->waitWorkComplete();
    }
  }
```

其有如下的几个特性

- 缓存复用：释放的块不立即释放给cuda，而是还到显存池中
- 大小对齐：向上取整到 2 的幂次，提高块复用率
- 预留段: 预分配一大块 pinned 内存（可配置 MB 数），小分配从中快速取
- 用 CUDA event 确认 GPU 不再使用后才复用块，这是因为torch的分配其实是pre stream的，需要去记录这个block被哪些stream所使用，保持不被意外释放

分配内存的路径如下所示，会首先去看保留下来的块，这个路径相对来说较快，假设没有可用的块，那么将会采用两种策略进行分配

- 策略 A（默认）：通过cudaHostAlloc进行分配
- 策略 B：malloc + cudaHostRegister

```shell
allocate_host_memory(size, ptr)
    │
    ├─ 1. Reserve Segment 快速路径（L43-48）
    │      如果 pinned_reserve_segment_size_mb > 0 且有剩余空间
    │      → 直接指针偏移，零 CUDA API 调用，直接返回
    │
    └─ 2. slowpath（L52-93）
           │
           ├─ 选 Device Context（L61-67）：找有 primary context 的设备
           │   （避免 NUMA 跨节点，注释 L53-60 解释了原因）
           │
           └─ pinned_use_cuda_host_register?（L70）
               │
               ├─ false（默认）→ cudaHostAlloc()  策略 A
               │
               └─ true → allocWithCudaHostRegister()  策略 B
                          ├─ malloc()
                          ├─ 并行 page pre-fault（L208-243）
                          │   条件：numMapThreads > 1 且 size ≥ pageSize × numThreads
                          └─ cudaHostRegister()
```

相关配置项
通过 PYTORCH_CUDA_ALLOC_CONF 设置：


开启策略 B，并用 8 线程并行 pre-fault
```shell
PYTORCH_CUDA_ALLOC_CONF=pinned_use_cuda_host_register:True,pinned_num_register_threads:8
```

预留 512MB reserve segment（绕过 slowpath 的快速路径）

```shell
PYTORCH_CUDA_ALLOC_CONF=pinned_reserve_segment_size_mb:512
m_pinned_num_register_threads = 1
```
