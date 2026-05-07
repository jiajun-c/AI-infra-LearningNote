# CUDA 虚拟内存管理 (VMM)

CUDA VMM（Virtual Memory Management）API 于 CUDA 10.2 引入，允许开发者显式控制**物理内存分配**与**虚拟地址映射**的绑定关系，解决了传统 `cudaMalloc` 无法满足的几类场景。

## 核心概念

| 概念 | 说明 |
|------|------|
| 物理内存块 (Physical Chunk) | 由 `cuMemCreate` 分配，独立于虚拟地址 |
| 虚拟地址空间 (VA Range) | 由 `cuMemAddressReserve` 预留，不消耗物理内存 |
| 映射 (Mapping) | `cuMemMap` 将物理块绑定到某段虚拟地址 |
| 访问权限 | `cuMemSetAccess` 控制哪些设备可读写该映射 |

## 与 cudaMalloc 的对比

```
cudaMalloc:  [PA] ←1:1绑定→ [VA]   分配即映射，无法解耦
VMM:         [PA]            [VA]   独立管理，按需映射
```

## 主要优势

1. **动态扩展无需 memcpy**：预留大段虚拟地址，按需提交物理内存，指针不变
2. **多设备 P2P 共享**：同一块物理内存可同时映射到多个 GPU 的虚拟地址空间
3. **显存碎片控制**：可将不连续的物理块映射为连续虚拟地址
4. **延迟物理分配**：预留 VA 不占用显存，只有 `cuMemMap` 后才消耗物理内存

## API 速查

```c
CUmemAllocationProp prop = {}; // 属性结构体
prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED; // 分配到pin memory
// typedef enum CUmemLocationType_enum {
//     CU_MEM_LOCATION_TYPE_INVALID    = 0x0,
//     CU_MEM_LOCATION_TYPE_DEVICE     = 0x1,  /**< Location is a device location, thus id is a device ordinal */
//     CU_MEM_LOCATION_TYPE_HOST       = 0x2,   /**< Location is host, id is ignored */
//     CU_MEM_LOCATION_TYPE_HOST_NUMA  = 0x3,  /**< Location is a host NUMA node, thus id is a host NUMA node id */
//     CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT = 0x4,  /**< Location is a host NUMA node of the current thread, id is ignored */
//     CU_MEM_LOCATION_TYPE_MAX        = 0x7FFFFFFF
// } CUmemLocationType;
prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;   // 分配在GPU设备上，所以下面的id是device
prop.location.id   = device;                        // device id

// 查询分配粒度（物理块大小必须对齐到此值）
cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
// typedef enum CUmemAllocationGranularity_flags_enum {
//     CU_MEM_ALLOC_GRANULARITY_MINIMUM     = 0x0,     /**< Minimum required granularity for allocation */
//     CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1      /**< Recommended granularity for allocation for best performance */
// } CUmemAllocationGranularity_flags;

// 分配物理内存块（不与任何 VA 绑定）
cuMemCreate(&handle, size, &prop, 0);

// 预留虚拟地址空间（不消耗物理内存）
cuMemAddressReserve(&va, size, alignment, hint, 0);

// 映射物理块到虚拟地址
cuMemMap(va, size, 0, handle, 0);

// 设置访问权限
cuMemSetAccess(va, size, &accessDesc, 1);

// 释放（顺序：Unmap → Free VA → Release physical）
cuMemUnmap(va, size);
cuMemAddressFree(va, size);
cuMemRelease(handle);
```

## Demo 说明

[vmm_demo.cu](vmm_demo.cu) 包含三个递进的示例：

### Demo 1：基础 VMM
演示最基本的物理内存分配、VA 预留、映射、权限设置和释放流程。将两块物理内存映射到连续虚拟地址。

### Demo 2：动态可扩展数组
预留 64 MB 虚拟地址空间但初始不提交物理内存，之后两次 `grow` 各提交 8 MB。
关键点：扩展时 **指针不变**，已有数据完全保留，无需 `memcpy`。

```
VA: [0 ---- 8MB ---- 16MB ---- ... ---- 64MB]
         ↑已映射物理内存  ↑按需扩展
```

### Demo 3：多设备 P2P 映射
同一块物理内存同时映射到两张 GPU 的不同虚拟地址，实现零拷贝 P2P 共享。需要两张支持 P2P 的 GPU。

## 编译运行

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
./vmm_demo
```

要求：
- CUDA 10.2+，驱动 440+
- Pascal (sm_60) 或更新架构
- `CMakeLists.txt` 中的 `CMAKE_CUDA_ARCHITECTURES` 改为实际 GPU 的 sm 版本

## 注意事项

- VMM API 使用驱动 API（`libcuda.so`），不是运行时 API（`libcudart.so`）
- 释放顺序严格：`cuMemUnmap` → `cuMemAddressFree` → `cuMemRelease`
- `cuMemRelease` 后物理内存仍被已有映射引用，驱动通过引用计数管理
- 分配粒度通常为 2 MB，物理块大小必须是它的整数倍
