/**
 * CUDA 虚拟内存管理 (VMM) Demo
 *
 * 演示 CUDA VMM API 的核心功能：
 * 1. 分配物理内存块 (cuMemCreate)
 * 2. 预留虚拟地址空间 (cuMemAddressReserve)
 * 3. 映射物理内存到虚拟地址 (cuMemMap)
 * 4. 设置访问权限 (cuMemSetAccess)
 * 5. 动态扩展内存（无需 memcpy 的动态数组）
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        CUresult err = (call);                                                  \
        if (err != CUDA_SUCCESS) {                                              \
            const char *errStr;                                                 \
            cuGetErrorString(err, &errStr);                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    errStr);                                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ============================================================
// Demo 1: 基础 VMM — 手动管理物理块与虚拟地址
// ============================================================
void demo_basic_vmm(int device) {
    printf("\n=== Demo 1: 基础 VMM ===\n");

    size_t granularity = 0;
    CUmemAllocationProp prop = {};
    prop.type             = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type    = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id      = device;

    // 查询分配粒度（物理块大小必须是它的整数倍）
    CHECK_CUDA(cuMemGetAllocationGranularity(&granularity, &prop,
                                             CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    printf("分配粒度: %zu bytes (%zu MB)\n", granularity,
           granularity / (1024 * 1024));

    // 分配 2 个物理块，每块 granularity 大小
    size_t chunk_size = granularity;
    CUmemGenericAllocationHandle handles[2];
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cuMemCreate(&handles[i], chunk_size, &prop, 0));
    }

    // 预留连续虚拟地址空间（2 * chunk_size）
    CUdeviceptr va_base = 0;
    size_t va_size = 2 * chunk_size;
    CHECK_CUDA(cuMemAddressReserve(&va_base, va_size, 0 /*align*/, 0, 0));
    printf("预留虚拟地址: 0x%llx，大小: %zu bytes\n",
           (unsigned long long)va_base, va_size);

    // 将两块物理内存分别映射到虚拟地址的前半段和后半段
    CHECK_CUDA(cuMemMap(va_base,              chunk_size, 0, handles[0], 0));
    CHECK_CUDA(cuMemMap(va_base + chunk_size, chunk_size, 0, handles[1], 0));

    // 设置访问权限（当前设备可读写）
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id   = device;
    access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CUDA(cuMemSetAccess(va_base, va_size, &access, 1));

    // 使用指针写入数据（通过 cuMemsetD32 初始化）
    CHECK_CUDA(cuMemsetD32(va_base, 0xDEADBEEF, va_size / sizeof(int)));

    // 读回验证
    int first_val = 0;
    CHECK_CUDA(cuMemcpyDtoH(&first_val, va_base, sizeof(int)));
    printf("写入并读回第一个 int: 0x%X (期望 0xDEADBEEF)\n", first_val);
    assert((unsigned)first_val == 0xDEADBEEFU);

    // 释放：顺序必须是 Unmap -> Release VA -> Free physical
    CHECK_CUDA(cuMemUnmap(va_base,              chunk_size));
    CHECK_CUDA(cuMemUnmap(va_base + chunk_size, chunk_size));
    CHECK_CUDA(cuMemAddressFree(va_base, va_size));
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cuMemRelease(handles[i]));
    }
    printf("Demo 1 完成\n");
}

// ============================================================
// Demo 2: 动态可扩展数组（无需 memcpy）
// ============================================================

typedef struct {
    CUdeviceptr ptr;        // 当前虚拟地址起点
    size_t      va_size;    // 已预留的虚拟地址空间
    size_t      committed;  // 已映射（已提交）的物理内存大小
    size_t      granularity;
    int         device;
    CUmemAllocationProp prop;
} DynArray;

static void dynarray_init(DynArray *da, int device, size_t initial_va_reserve) {
    da->device = device;
    da->prop   = (CUmemAllocationProp){};
    da->prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    da->prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    da->prop.location.id   = device;

    CHECK_CUDA(cuMemGetAllocationGranularity(&da->granularity, &da->prop,
                                              CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    // 对齐到粒度
    da->va_size = ((initial_va_reserve + da->granularity - 1)
                   / da->granularity) * da->granularity;
    da->committed = 0;

    CHECK_CUDA(cuMemAddressReserve(&da->ptr, da->va_size, 0, 0, 0));
    printf("动态数组：预留 VA %zu bytes @ 0x%llx\n", da->va_size,
           (unsigned long long)da->ptr);
}

// 提交更多物理内存（原地扩展，不移动指针）
static void dynarray_grow(DynArray *da, size_t extra_bytes) {
    size_t aligned = ((extra_bytes + da->granularity - 1)
                      / da->granularity) * da->granularity;

    if (da->committed + aligned > da->va_size) {
        fprintf(stderr, "超出预留的虚拟地址空间！\n");
        exit(1);
    }

    CUmemGenericAllocationHandle handle;
    CHECK_CUDA(cuMemCreate(&handle, aligned, &da->prop, 0));
    CHECK_CUDA(cuMemMap(da->ptr + da->committed, aligned, 0, handle, 0));

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id   = da->device;
    access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CUDA(cuMemSetAccess(da->ptr + da->committed, aligned, &access, 1));

    // 初始化新提交的区域
    CHECK_CUDA(cuMemsetD32(da->ptr + da->committed, 0,
                           aligned / sizeof(int)));

    da->committed += aligned;
    printf("扩展 %zu bytes，当前已提交: %zu bytes\n", aligned, da->committed);

    // 注意：handle 释放后物理内存仍被映射，引用计数由驱动管理
    CHECK_CUDA(cuMemRelease(handle));
}

static void dynarray_destroy(DynArray *da) {
    if (da->committed > 0) {
        // 逐粒度 unmap（这里简化为一次性 unmap 整段已提交区域）
        CHECK_CUDA(cuMemUnmap(da->ptr, da->committed));
    }
    CHECK_CUDA(cuMemAddressFree(da->ptr, da->va_size));
    da->ptr       = 0;
    da->committed = 0;
}

void demo_dynamic_array(int device) {
    printf("\n=== Demo 2: 动态可扩展数组 ===\n");

    // 预留 64 MB 虚拟地址空间，但初始不提交物理内存
    DynArray da;
    dynarray_init(&da, device, 64ULL * 1024 * 1024);

    // 第一次扩展：提交 8 MB
    dynarray_grow(&da, 8ULL * 1024 * 1024);

    // 写入数据
    size_t n = 1024 * 1024; // 1M 个 int
    CHECK_CUDA(cuMemsetD32(da.ptr, 42, n));

    // 再次扩展：再提交 8 MB，数据不会移动
    dynarray_grow(&da, 8ULL * 1024 * 1024);

    // 原来的数据仍然有效
    int val = 0;
    CHECK_CUDA(cuMemcpyDtoH(&val, da.ptr, sizeof(int)));
    printf("扩展后原数据: %d (期望 42)\n", val);
    assert(val == 42);

    dynarray_destroy(&da);
    printf("Demo 2 完成\n");
}

// ============================================================
// Demo 3: 多设备 P2P 映射同一块物理内存
// ============================================================
void demo_p2p_mapping(int dev0, int dev1) {
    printf("\n=== Demo 3: P2P 多设备映射 ===\n");

    // 检查 P2P 访问能力
    int can_access = 0;
    cuDeviceCanAccessPeer(&can_access, dev0, dev1);
    if (!can_access) {
        printf("设备 %d 无法 P2P 访问设备 %d，跳过 Demo 3\n", dev0, dev1);
        return;
    }

    size_t granularity = 0;
    CUmemAllocationProp prop = {};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = dev0;
    CHECK_CUDA(cuMemGetAllocationGranularity(&granularity, &prop,
                                              CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    // 在 dev0 上分配物理内存
    CUmemGenericAllocationHandle handle;
    CHECK_CUDA(cuMemCreate(&handle, granularity, &prop, 0));

    // dev0 的虚拟映射
    CUdeviceptr va0 = 0;
    CHECK_CUDA(cuMemAddressReserve(&va0, granularity, 0, 0, 0));
    CHECK_CUDA(cuMemMap(va0, granularity, 0, handle, 0));

    CUmemAccessDesc access[2] = {};
    access[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access[0].location.id   = dev0;
    access[0].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access[1].location.id   = dev1;
    access[1].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CUDA(cuMemSetAccess(va0, granularity, access, 2));

    // dev1 也映射同一块物理内存
    CUdeviceptr va1 = 0;
    CHECK_CUDA(cuMemAddressReserve(&va1, granularity, 0, 0, 0));
    CHECK_CUDA(cuMemMap(va1, granularity, 0, handle, 0));
    CHECK_CUDA(cuMemSetAccess(va1, granularity, access, 2));

    printf("Dev0 VA: 0x%llx  Dev1 VA: 0x%llx 映射到同一物理块\n",
           (unsigned long long)va0, (unsigned long long)va1);

    // dev0 写，dev1 读
    CHECK_CUDA(cuMemsetD32(va0, 0xCAFEBABE, granularity / sizeof(int)));
    int val = 0;
    CHECK_CUDA(cuMemcpyDtoH(&val, va1, sizeof(int)));
    printf("Dev0 写 0xCAFEBABE，Dev1 读回: 0x%X\n", (unsigned)val);
    assert((unsigned)val == 0xCAFEBABEU);

    CHECK_CUDA(cuMemUnmap(va0, granularity));
    CHECK_CUDA(cuMemUnmap(va1, granularity));
    CHECK_CUDA(cuMemAddressFree(va0, granularity));
    CHECK_CUDA(cuMemAddressFree(va1, granularity));
    CHECK_CUDA(cuMemRelease(handle));
    printf("Demo 3 完成\n");
}

// ============================================================
// main
// ============================================================
int main() {
    CHECK_CUDA(cuInit(0));

    int device_count = 0;
    CHECK_CUDA(cuDeviceGetCount(&device_count));
    printf("检测到 %d 个 CUDA 设备\n", device_count);

    CUdevice dev0;
    CHECK_CUDA(cuDeviceGet(&dev0, 0));

    // 检查 VMM 支持
    int vmm_supported = 0;
    CHECK_CUDA(cuDeviceGetAttribute(&vmm_supported,
                                    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
                                    dev0));
    if (!vmm_supported) {
        printf("当前设备不支持 VMM，需要 Pascal (sm_60) 或更新架构\n");
        return 1;
    }
    printf("VMM 支持: 是\n");

    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev0));

    demo_basic_vmm(0);
    demo_dynamic_array(0);

    if (device_count >= 2) {
        demo_p2p_mapping(0, 1);
    } else {
        printf("\n=== Demo 3: 跳过（需要 2 张 GPU）===\n");
    }

    CHECK_CUDA(cuCtxDestroy(ctx));
    printf("\n全部 Demo 完成\n");
    return 0;
}
