# CUDA 启动配置

## 1. Grid 和 Block 配置

在 CUDA 中，kernel 的启动需要配置 grid 和 block 的大小。

### 1.1 基本启动语法

```cpp
kernelName<<<gridDim, blockDim, sharedMemSize, stream>>>(arguments);
```

### 1.2 GridDim 配置

### 1.3 BlockDim 配置

### 1.4 共享内存配置

### 1.5 Stream 配置

## 2. 最佳实践

## 3. 常见错误
