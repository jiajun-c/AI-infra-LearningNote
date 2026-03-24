# CUDA JIT编译 (NVRTC)

## 什么是JIT编译

JIT (Just-In-Time) 编译是在**程序运行时**动态编译代码的技术，区别于传统的AOT (Ahead-Of-Time) 预编译方式。

NVIDIA 提供了 **NVRTC** (NVIDIA Runtime Compilation) 库，允许在运行时将CUDA C++源码字符串编译为PTX，再通过CUDA Driver API加载执行。

## JIT vs AOT 核心对比

| 特性 | AOT (`nvcc`编译) | JIT (NVRTC运行时编译) |
|------|-----------------|---------------------|
| 编译时机 | 构建时 | 运行时 |
| 内核特化 | 模板参数必须编译时已知，需穷举 | 运行时值可作为编译时常量注入 |
| GPU架构适配 | `-arch`在构建时固定 | 运行时检测GPU，自动适配 |
| 代码生成 | 源码在构建前就固定不变 | 可动态拼接、按需生成源码 |
| 分发部署 | 需为不同GPU预编译多个binary | 单一binary，运行时编译适配 |
| 编译开销 | 构建时一次性完成 | 运行时有编译延迟（毫秒级） |

## JIT的三大核心优势

### 1. 运行时特化 (Runtime Specialization)

AOT编译时，`BLOCK_SIZE`要么是C++模板参数（需穷举所有可能值），要么是函数参数（编译器无法优化循环展开）。

JIT可以在运行时查询GPU的`maxThreadsPerBlock`，选择最优值，然后将其作为字面量常量注入kernel源码 → 编译器完全展开循环。

```cpp
// AOT方式: blockSize是运行时参数，编译器无法展开循环
__global__ void reduce(float* data, int blockSize_param) {
    for (int s = blockSize_param / 2; s > 0; s >>= 1) { ... }
}

// JIT方式: BLOCK_SIZE=256 是字面量常量，编译器完全展开为8条指令
// (这段源码是运行时动态生成的字符串)
__global__ void reduce(float* data) {
    for (int s = 128; s > 0; s >>= 1) { ... }  // 编译器展开!
}
```

### 2. 动态代码生成 (Dynamic Code Generation)

根据运行时条件（用户输入、配置文件等）组装不同的kernel源码。例如同一个reduce模板，运行时决定是sum/max/min操作，生成各自独立的kernel，内联最优操作，消除运行时分支。

### 3. 架构自适应 (Architecture Adaptation)

运行时检测GPU计算能力，传递 `--gpu-architecture=compute_XX` 给NVRTC。同一程序在H100上编译出sm_90的PTX，在A100上编译出sm_80的PTX，无需预编译多个版本。

## NVRTC编译流程

```
CUDA源码字符串 (运行时动态生成)
    │
    ▼
nvrtcCreateProgram()      ← 创建编译程序对象
    │
    ▼
nvrtcCompileProgram()     ← 编译 (传入 --gpu-architecture 等选项)
    │
    ▼
nvrtcGetPTX()             ← 获取编译产物 (PTX中间码)
    │
    ▼
cuModuleLoadDataEx()      ← CUDA Driver API 加载PTX为module
    │
    ▼
cuModuleGetFunction()     ← 从module中获取kernel函数句柄
    │
    ▼
cuLaunchKernel()          ← 启动kernel执行
```

## 本示例的三组实验

### 实验1: 运行时特化 vs AOT通用Kernel

- **JIT版**: `BLOCK_SIZE` 作为编译时常量注入 → 循环完全展开，寄存器优化
- **AOT版**: `blockSize` 作为函数参数传入 → 编译器无法展开循环
- 对比两者执行性能差异

### 实验2: 动态代码生成

- 同一模板函数 `generateKernelSource(blockSize, opType)` 根据参数生成不同kernel
- 运行时生成 sum / max / min 三种归约操作的独立kernel
- 每个kernel内联对应操作，无运行时分支判断

### 实验3: 架构自适应

- 运行时检测GPU计算能力 (sm_XX)
- 展示生成的PTX中 `.target` 指令，证明kernel是针对当前GPU架构编译

## 编译与运行

```bash
make        # 编译
make run    # 运行
make clean  # 清理
```

## 注意事项

- NVRTC编译有一定开销（通常毫秒级），适合kernel需要多次执行的场景
- 生产环境中可缓存编译后的PTX，避免重复编译
- NVRTC编译的kernel需通过CUDA Driver API (`cuLaunchKernel`) 启动，不能使用Runtime API的 `<<<>>>` 语法
- NVRTC仅编译device代码，host代码仍需AOT编译

## 参考

- [NVRTC官方文档](https://docs.nvidia.com/cuda/nvrtc/)
- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)
