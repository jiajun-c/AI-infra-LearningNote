# CUDA IPC 通信

CUDA IPC 通信指的是一个进程创建的 GPU 内存或者 GPU event 可以被另外一个进程打开并使用。

## 1. The use case

当一个进程里的 GPU 想访问另一个进程创建的 GPU 显存时，可以用 CUDA IPC 导出/打开显存 handle。
如果两个 GPU 之间支持 P2P，那么打开 handle 的进程可以直接从本 GPU kernel 访问对端 GPU 显存，数据路径通常走 NVLink/NVSwitch 或 PCIe P2P，而不是先拷贝到 CPU host memory。

常见场景：

- 多进程训练或推理中，不同进程共享 GPU buffer。
- 一个进程负责生产数据，另一个进程负责消费数据。
- 需要跨进程验证 GPU-GPU P2P 访问路径。

## 2. 核心 API

导出进程：

```cpp
cudaMalloc(&d_ptr, bytes);
cudaIpcGetMemHandle(&handle, d_ptr);
```

打开进程：

```cpp
cudaIpcOpenMemHandle(
    reinterpret_cast<void**>(&remote_ptr),
    handle,
    cudaIpcMemLazyEnablePeerAccess
);
```

关闭映射：

```cpp
cudaIpcCloseMemHandle(remote_ptr);
```

其中 `cudaIpcMemLazyEnablePeerAccess` 会在需要时尝试启用 peer access。

## 3. Demo

本目录提供了一个 `fork + pipe` 的最小 demo：

- 先 `fork` 出 parent/child，避免 child 继承 parent 已经初始化过的 CUDA runtime 状态。
- parent process 使用 GPU0 `cudaMalloc` 一段显存，并初始化为 `3.14f`。
- parent 通过 `cudaIpcGetMemHandle` 导出 IPC handle。
- child process 使用 GPU1 打开这个 handle。
- child 在 GPU1 上启动 kernel，直接读取 GPU0 显存并写入 GPU1 本地输出。

源码：

```text
cuda_ipc_p2p_demo.cu
```

编译：

```bash
make
```

运行：

```bash
./cuda_ipc_p2p_demo
```

也可以手动编译：

```bash
nvcc -O2 -std=c++17 cuda_ipc_p2p_demo.cu -o cuda_ipc_p2p_demo
```

运行前可以先看 GPU 拓扑：

```bash
nvidia-smi topo -m
```

如果 GPU0 和 GPU1 之间显示 `NV1/NV2/NVL`，通常走 NVLink/NVSwitch；如果显示 `PIX/PXB/PHB`，通常走 PCIe；如果是 `SYS`，可能跨 CPU socket，P2P 性能会更差，也可能不可用。

## 4. 注意事项

- `cudaIpcGetMemHandle` 只能用于 `cudaMalloc` 分配的 device memory。
- 导出显存的进程必须在其他进程关闭 handle 前保持 alive，不能提前 `cudaFree`。
- CUDA IPC 只适用于同一台机器上的进程，不能跨节点。
- 是否真的能 P2P 访问取决于 GPU 拓扑、驱动和运行环境。
- 如果 `cudaDeviceCanAccessPeer(&can_access, dst_gpu, src_gpu)` 返回 0，跨 GPU 直接访问通常不可用。
