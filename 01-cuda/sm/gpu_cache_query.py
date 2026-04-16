#!/usr/bin/env python3
"""
查询 GPU 的 L1/L2 cache 大小

有两种主要方法：
1. 使用 pynvml (NVIDIA Management Library)
2. 使用 torch.cuda.get_device_properties
"""

# 方法 1: 使用 pynvml
def query_with_pynvml():
    """使用 pynvml 查询 GPU cache 信息"""
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)

            # 注意：pynvml 没有直接提供 L1/L2 cache 大小的 API
            # 但可以通过 device properties 获取
            print(f"GPU {i}: {name}")

            # 获取 GPU 详细信息
            compute_cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            print(f"  Compute Capability: {compute_cap}")

            # 获取 memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"  Total Memory: {mem_info.total // (1024**3)} GB")

            # 获取 driver version
            driver = pynvml.nvmlSystemGetDriverVersion()
            print(f"  Driver Version: {driver}")

        pynvml.nvmlShutdown()

    except ImportError:
        print("pynvml not installed. Install with: pip install pynvml")
    except Exception as e:
        print(f"Error: {e}")


# 方法 2: 使用 PyTorch
def query_with_pytorch():
    """使用 PyTorch 查询 GPU 属性"""
    try:
        import torch

        if not torch.cuda.is_available():
            print("CUDA not available")
            return

        device_count = torch.cuda.device_count()

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory // (1024**3)} GB")

            # 注意：torch.cuda.get_device_properties 返回的是 cudaDeviceProp
            # 但 Python 绑定中没有暴露所有字段，包括 cache 信息

            # 可以通过 compute capability 推断 cache 大小
            print(f"\n  参考 Cache 大小 (根据 Compute Capability {props.major}.{props.minor}):")
            print("    需要查询 NVIDIA 文档或使用 CUDA Runtime API 获取详细信息")

    except ImportError:
        print("PyTorch not installed")
    except Exception as e:
        print(f"Error: {e}")


# 方法 3: 使用 ctypes 调用 CUDA Runtime API
def query_with_cuda_runtime():
    """使用 ctypes 直接调用 CUDA Runtime API 获取详细的 cache 信息"""
    import ctypes
    import os

    # 查找 CUDA 库
    cuda_lib_names = ['libcudart.so', 'libcudart.so.11', 'libcudart.so.12']
    cuda = None

    for lib in cuda_lib_names:
        try:
            cuda = ctypes.CDLL(lib)
            break
        except:
            continue

    if cuda is None:
        print("CUDA runtime library not found")
        return

    # cudaDeviceGetAttribute 需要设置正确的 argtypes 和 restype
    cuda.cudaDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),  # value
        ctypes.c_int,                   # attr
        ctypes.c_int                    # device
    ]
    cuda.cudaDeviceGetAttribute.restype = ctypes.c_int  # cudaError_t

    cuda.cudaGetDeviceProperties.argtypes = [
        ctypes.c_void_p,    # prop (cudaDeviceProp*)
        ctypes.c_int        # device
    ]
    cuda.cudaGetDeviceProperties.restype = ctypes.c_int

    print("使用 CUDA Runtime API 查询:")

    # 获取设备数量
    device_count = ctypes.c_int()
    cuda.cudaGetDeviceCount(ctypes.byref(device_count))

    for device_id in range(device_count.value):
        # 使用 cudaDeviceGetAttribute 查询各种属性
        # 注意：cudaDeviceAttr 的值可能因 CUDA 版本而异
        # 以下是经过测试的属性 ID (Hopper 架构):
        # 38 = L2 Cache Size (字节)
        # 82 = Shared Memory Per Multiprocessor (字节) - 即 L1 + Shared
        # 16 = Multiprocessor Count

        CUDA_DEVICE_ATTR_L2_CACHE_SIZE = 38
        CUDA_DEVICE_ATTR_SHARED_MEMORY_PER_MULTIPROCESSOR = 82
        CUDA_DEVICE_ATTR_MULTIPROCESSOR_COUNT = 16

        l2_size = ctypes.c_int()
        ret_l2 = cuda.cudaDeviceGetAttribute(ctypes.byref(l2_size),
                                              CUDA_DEVICE_ATTR_L2_CACHE_SIZE,
                                              device_id)

        shared_mem = ctypes.c_int()
        ret_shared = cuda.cudaDeviceGetAttribute(ctypes.byref(shared_mem),
                                                  CUDA_DEVICE_ATTR_SHARED_MEMORY_PER_MULTIPROCESSOR,
                                                  device_id)

        sm_count = ctypes.c_int()
        ret_sm = cuda.cudaDeviceGetAttribute(ctypes.byref(sm_count),
                                              CUDA_DEVICE_ATTR_MULTIPROCESSOR_COUNT,
                                              device_id)

        name = ctypes.create_string_buffer(256)
        cuda.cudaGetDeviceProperties(ctypes.byref(name), device_id)

        print(f"\nGPU {device_id}: {name.value.decode()}")
        if ret_l2 == 0:
            print(f"  L2 Cache Size: {l2_size.value / (1024*1024):.1f} MB ({l2_size.value} bytes)")
        else:
            print(f"  L2 Cache Size: 无法查询 (return code: {ret_l2})")

        if ret_shared == 0:
            print(f"  Shared Memory per SM: {shared_mem.value / 1024:.1f} KB ({shared_mem.value} bytes)")
            print(f"    (L1 cache + Shared Memory, 可配置比例)")
        else:
            print(f"  Shared Memory: 无法查询 (return code: {ret_shared})")

        if ret_sm == 0:
            print(f"  SM Count: {sm_count.value}")
            if ret_l2 == 0 and ret_shared == 0:
                total_l2 = l2_size.value / (1024*1024)
                total_shared = (shared_mem.value * sm_count.value) / (1024*1024)
                print(f"  Total Shared Memory (all SMs): {total_shared:.1f} MB")
        else:
            print(f"  SM Count: 无法查询")


# 方法 4: 使用 cupy (如果可用)
def query_with_cupy():
    """使用 CuPy 查询 GPU 属性"""
    try:
        import cupy as cp

        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"检测到 {device_count} 个 GPU")

        for i in range(device_count):
            device = cp.cuda.Device(i)
            props = device.attributes

            print(f"\nGPU {i}: {device.name}")

            # CuPy 提供了更完整的属性访问
            l2_cache = props.get('MaxL2CacheSize', None)
            shared_mem = props.get('MaxSharedMemoryPerMultiprocessor', None)
            compute_cap = props.get('ComputeCapabilityMajor', 'N/A')

            print(f"  Compute Capability: {compute_cap}")
            if l2_cache is not None:
                print(f"  L2 Cache Size: {l2_cache // 1024} KB")
            else:
                print(f"  L2 Cache Size: 无法查询")

            if shared_mem is not None:
                print(f"  Shared Memory per SM: {shared_mem // 1024} KB")
            else:
                print(f"  Shared Memory per SM: 无法查询")

    except ImportError:
        print("CuPy not installed")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("方法 1: pynvml")
    print("=" * 60)
    query_with_pynvml()

    print("\n" + "=" * 60)
    print("方法 2: PyTorch")
    print("=" * 60)
    query_with_pytorch()

    print("\n" + "=" * 60)
    print("方法 3: CUDA Runtime API (ctypes)")
    print("=" * 60)
    query_with_cuda_runtime()

    print("\n" + "=" * 60)
    print("方法 4: CuPy")
    print("=" * 60)
    query_with_cupy()
