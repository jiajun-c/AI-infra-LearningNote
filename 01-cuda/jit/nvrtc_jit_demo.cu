/**
 * CUDA JIT编译 (NVRTC) 特性演示
 *
 * 本示例通过3组实验，充分展示JIT编译相比AOT编译的核心优势:
 *   实验1: 运行时特化 — BLOCK_SIZE作为编译时常量 vs 运行时参数
 *   实验2: 动态代码生成 — 同一模板生成sum/max/min三种kernel
 *   实验3: 架构自适应 — 运行时检测GPU架构，生成对应PTX
 *
 * 依赖: NVRTC (-lnvrtc), CUDA Driver API (-lcuda)
 */

#include <nvrtc.h>        // NVRTC: 运行时编译CUDA源码为PTX
#include <cuda.h>         // CUDA Driver API: 加载PTX、启动kernel
#include <cuda_runtime.h> // CUDA Runtime API: 用于AOT对照kernel
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <cassert>

// ============================================================
// 错误检查宏
// ============================================================
#define NVRTC_CHECK(call)                                                 \
    do {                                                                  \
        nvrtcResult res = (call);                                         \
        if (res != NVRTC_SUCCESS) {                                       \
            fprintf(stderr, "NVRTC错误 %s:%d: %s\n", __FILE__, __LINE__, \
                    nvrtcGetErrorString(res));                             \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

#define CU_CHECK(call)                                                       \
    do {                                                                     \
        CUresult res = (call);                                               \
        if (res != CUDA_SUCCESS) {                                           \
            const char* errStr;                                              \
            cuGetErrorString(res, &errStr);                                  \
            fprintf(stderr, "CUDA Driver错误 %s:%d: %s\n", __FILE__,        \
                    __LINE__, errStr);                                        \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t res = (call);                                            \
        if (res != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA Runtime错误 %s:%d: %s\n", __FILE__,       \
                    __LINE__, cudaGetErrorString(res));                       \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ============================================================
// AOT编译的通用reduce kernel（对照组）
// 注意: blockSize_param 是运行时参数，编译器无法展开循环
// ============================================================
__global__ void reduce_sum_aot(const float* input, float* output, int n,
                               int blockSize_param) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockSize_param + tid;

    // 加载数据到共享内存
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // blockSize_param 是运行时变量 → 编译器无法展开此循环!
    // 每次迭代都需要判断循环条件，无法做寄存器优化
    for (int s = blockSize_param / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================
// JIT Kernel 源码生成器
// 核心思想: 将运行时才知道的参数(BLOCK_SIZE、操作类型)
//          作为字面量常量注入源码，让编译器充分优化
// ============================================================
std::string generateKernelSource(int blockSize, const std::string& opType) {
    // 根据操作类型确定归约操作和初始值
    std::string reduceOp, identity;
    if (opType == "sum") {
        reduceOp = "sdata[tid] + sdata[tid + s]";
        identity = "0.0f";
    } else if (opType == "max") {
        reduceOp = "fmaxf(sdata[tid], sdata[tid + s])";
        identity = "-INFINITY";
    } else if (opType == "min") {
        reduceOp = "fminf(sdata[tid], sdata[tid + s])";
        identity = "INFINITY";
    } else {
        fprintf(stderr, "不支持的操作类型: %s\n", opType.c_str());
        exit(1);
    }

    // 构造 kernel 源码字符串
    // 关键优势:
    //   1. BLOCK_SIZE 是字面量常量(如256)，不是变量
    //      → 编译器可以完全展开for循环(log2(256)=8次迭代)
    //      → 共享内存大小在编译时已知，优化内存访问
    //   2. 归约操作直接内联，无switch/if分支
    //      → 消除分支预测开销
    // NVRTC环境没有<cmath>，需手动定义INFINITY
    std::string source = R"(
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif
extern "C" __global__
void reduce_kernel(const float* __restrict__ input,
                   float* __restrict__ output, int n) {
    __shared__ float sdata[)" + std::to_string(blockSize) + R"(];  // BLOCK_SIZE=)" + std::to_string(blockSize) + R"( (编译时常量!)

    int tid = threadIdx.x;
    int idx = blockIdx.x * )" + std::to_string(blockSize) + R"( + tid;

    // 加载数据，越界用identity值填充
    sdata[tid] = (idx < n) ? input[idx] : )" + identity + R"(;
    __syncthreads();

    // 归约循环 — BLOCK_SIZE是字面量，编译器完全展开此循环
    // 展开后变为8条连续的if+操作，无循环开销
    #pragma unroll
    for (int s = )" + std::to_string(blockSize / 2) + R"(; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = )" + reduceOp + R"(;  // 操作: )" + opType + R"(
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}
)";
    return source;
}

// ============================================================
// NVRTC编译: 源码字符串 → PTX
// ============================================================
void compileWithNVRTC(const std::string& source, int smMajor, int smMinor,
                      std::vector<char>& ptx, std::string& compileLog) {
    // 1. 创建NVRTC程序对象
    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, source.c_str(),
                                   "reduce_kernel.cu",  // 虚拟文件名
                                   0, NULL, NULL));

    // 2. 设置编译选项
    //    关键: --gpu-architecture 在运行时确定，适配当前GPU!
    char archOpt[64];
    snprintf(archOpt, sizeof(archOpt), "--gpu-architecture=compute_%d%d",
             smMajor, smMinor);
    const char* opts[] = {archOpt, "--use_fast_math"};

    // 3. 编译
    nvrtcResult compileRes = nvrtcCompileProgram(prog, 2, opts);

    // 4. 获取编译日志（即使编译成功也可能有警告信息）
    size_t logSize;
    NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
    compileLog.resize(logSize);
    NVRTC_CHECK(nvrtcGetProgramLog(prog, &compileLog[0]));

    if (compileRes != NVRTC_SUCCESS) {
        fprintf(stderr, "NVRTC编译失败!\n编译日志:\n%s\n", compileLog.c_str());
        nvrtcDestroyProgram(&prog);
        exit(1);
    }

    // 5. 获取编译产物 PTX
    size_t ptxSize;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
    ptx.resize(ptxSize);
    NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));

    NVRTC_CHECK(nvrtcDestroyProgram(&prog));
}

// ============================================================
// 从PTX加载kernel函数 (CUDA Driver API)
// ============================================================
CUfunction loadKernelFromPTX(const std::vector<char>& ptx,
                             const char* kernelName) {
    CUmodule module;
    CU_CHECK(cuModuleLoadDataEx(&module, ptx.data(), 0, NULL, NULL));

    CUfunction func;
    CU_CHECK(cuModuleGetFunction(&func, module, kernelName));
    return func;
}

// ============================================================
// JIT kernel 性能测试 (CUDA Driver API: cuLaunchKernel)
// ============================================================
float benchmarkJIT(CUfunction kernel, CUdeviceptr d_input,
                   CUdeviceptr d_output, int n, int blockSize, int numRuns) {
    int gridSize = (n + blockSize - 1) / blockSize;
    void* args[] = {&d_input, &d_output, &n};

    // 预热
    CU_CHECK(cuLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1, 1, 0, 0,
                            args, NULL));
    CU_CHECK(cuCtxSynchronize());

    // 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < numRuns; i++) {
        CU_CHECK(cuLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1, 1, 0,
                                0, args, NULL));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / numRuns;
}

// ============================================================
// AOT kernel 性能测试 (CUDA Runtime API: <<<>>>)
// ============================================================
float benchmarkAOT(const float* d_input, float* d_output, int n,
                   int blockSize, int numRuns) {
    int gridSize = (n + blockSize - 1) / blockSize;
    size_t smemSize = blockSize * sizeof(float);

    // 预热
    reduce_sum_aot<<<gridSize, blockSize, smemSize>>>(d_input, d_output, n,
                                                      blockSize);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < numRuns; i++) {
        reduce_sum_aot<<<gridSize, blockSize, smemSize>>>(d_input, d_output, n,
                                                          blockSize);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / numRuns;
}

// ============================================================
// CPU参考实现 (用于结果验证)
// ============================================================
float cpuReduce(const float* data, int n, const std::string& opType) {
    float result;
    if (opType == "sum") {
        result = 0.0f;
        for (int i = 0; i < n; i++) result += data[i];
    } else if (opType == "max") {
        result = -INFINITY;
        for (int i = 0; i < n; i++)
            result = (data[i] > result) ? data[i] : result;
    } else {  // min
        result = INFINITY;
        for (int i = 0; i < n; i++)
            result = (data[i] < result) ? data[i] : result;
    }
    return result;
}

// ============================================================
// main — 三组实验
// ============================================================
int main() {
    printf("========================================\n");
    printf("  CUDA JIT编译 (NVRTC) 特性演示\n");
    printf("========================================\n\n");

    // ---------------------------------------------------
    // 1. 初始化 CUDA Driver API
    // ---------------------------------------------------
    CU_CHECK(cuInit(0));
    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, 0));
    CUcontext context;
    CU_CHECK(cuCtxCreate(&context, 0, device));

    // ---------------------------------------------------
    // 2. 运行时查询GPU属性 (JIT的基础：根据硬件信息做决策)
    // ---------------------------------------------------
    int smMajor, smMinor, maxThreads;
    CU_CHECK(cuDeviceGetAttribute(
        &smMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CU_CHECK(cuDeviceGetAttribute(
        &smMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    CU_CHECK(cuDeviceGetAttribute(
        &maxThreads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));

    char deviceName[256];
    CU_CHECK(cuDeviceGetName(deviceName, sizeof(deviceName), device));
    printf("[GPU信息] %s\n", deviceName);
    printf("[GPU信息] 计算能力: sm_%d%d, 最大线程数/块: %d\n\n", smMajor,
           smMinor, maxThreads);

    // ---------------------------------------------------
    // 3. 运行时决定最优 BLOCK_SIZE
    //    AOT编译时这个值必须硬编码或作为模板参数穷举
    //    JIT可以在运行时选择最优值，再作为编译时常量注入!
    // ---------------------------------------------------
    int blockSize = 256;
    if (maxThreads < blockSize) blockSize = maxThreads;
    printf("[运行时决策] 选择 BLOCK_SIZE = %d\n", blockSize);
    printf("  → AOT方式: 只能用模板穷举(template<int N>)或运行时参数\n");
    printf("  → JIT方式: 直接将%d作为字面量常量注入kernel源码\n\n", blockSize);

    // ---------------------------------------------------
    // 4. 准备测试数据
    // ---------------------------------------------------
    const int N = 1 << 22;  // 4M 元素
    size_t bytes = N * sizeof(float);
    std::vector<float> h_input(N);
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 1000) / 1000.0f;
    }

    // Driver API 内存 (给JIT kernel用)
    CUdeviceptr d_input_drv, d_output_jit;
    int gridSize = (N + blockSize - 1) / blockSize;
    CU_CHECK(cuMemAlloc(&d_input_drv, bytes));
    CU_CHECK(cuMemAlloc(&d_output_jit, gridSize * sizeof(float)));
    CU_CHECK(cuMemcpyHtoD(d_input_drv, h_input.data(), bytes));

    // Runtime API 内存 (给AOT kernel用)
    float *d_input_rt, *d_output_aot;
    CUDA_CHECK(cudaMalloc(&d_input_rt, bytes));
    CUDA_CHECK(cudaMalloc(&d_output_aot, gridSize * sizeof(float)));
    CUDA_CHECK(
        cudaMemcpy(d_input_rt, h_input.data(), bytes, cudaMemcpyHostToDevice));

    // ====================================================
    // 实验1: 运行时特化 vs AOT通用Kernel
    // ====================================================
    printf("===================================================\n");
    printf("  实验1: 运行时特化 vs AOT通用Kernel (性能对比)\n");
    printf("===================================================\n");
    printf("JIT版: BLOCK_SIZE=%d 作为编译时常量 → 循环完全展开\n", blockSize);
    printf("AOT版: blockSize=%d 作为函数参数   → 编译器无法优化\n\n",
           blockSize);

    // 生成JIT kernel源码
    std::string srcSum = generateKernelSource(blockSize, "sum");
    printf("--- JIT动态生成的Kernel源码 ---\n%s\n", srcSum.c_str());

    // NVRTC编译
    std::vector<char> ptxSum;
    std::string logSum;
    compileWithNVRTC(srcSum, smMajor, smMinor, ptxSum, logSum);
    printf("NVRTC编译成功! PTX大小: %zu 字节\n", ptxSum.size());
    if (logSum.length() > 1) {
        printf("编译日志: %s\n", logSum.c_str());
    }

    // 加载JIT kernel
    CUfunction jitSumKernel = loadKernelFromPTX(ptxSum, "reduce_kernel");
    printf("Kernel加载成功!\n\n");

    // 性能对比
    const int NUM_RUNS = 200;
    printf("[Benchmark] 数据量: %d 元素 (%.1f MB), 运行 %d 次取平均\n", N,
           (float)bytes / 1024 / 1024, NUM_RUNS);

    float jitTime =
        benchmarkJIT(jitSumKernel, d_input_drv, d_output_jit, N, blockSize,
                     NUM_RUNS);
    float aotTime =
        benchmarkAOT(d_input_rt, d_output_aot, N, blockSize, NUM_RUNS);

    printf("  JIT特化Kernel: %.4f ms\n", jitTime);
    printf("  AOT通用Kernel: %.4f ms\n", aotTime);
    printf("  加速比: %.2fx\n", aotTime / jitTime);

    // 验证正确性
    float jitResult;
    CU_CHECK(cuMemcpyDtoH(&jitResult, d_output_jit, sizeof(float)));
    float cpuResult = cpuReduce(h_input.data(), blockSize, "sum");
    printf("  正确性验证(首block): JIT=%.4f, CPU=%.4f, 误差=%.6f\n\n",
           jitResult, cpuResult, fabsf(jitResult - cpuResult));

    // ====================================================
    // 实验2: 动态代码生成 — 不同归约操作
    // ====================================================
    printf("===================================================\n");
    printf("  实验2: 动态代码生成 — 运行时决定操作类型\n");
    printf("===================================================\n");
    printf("同一模板，根据运行时参数生成不同操作的kernel\n");
    printf("→ 无需switch/if分支，每个kernel内联最优操作\n\n");

    const char* ops[] = {"sum", "max", "min"};
    const char* opNames[] = {"求和(sum)", "最大值(max)", "最小值(min)"};

    for (int i = 0; i < 3; i++) {
        // 为每种操作动态生成独立的kernel
        std::string src = generateKernelSource(blockSize, ops[i]);
        std::vector<char> ptx;
        std::string log;
        compileWithNVRTC(src, smMajor, smMinor, ptx, log);
        CUfunction kernel = loadKernelFromPTX(ptx, "reduce_kernel");

        // 性能测试
        float time = benchmarkJIT(kernel, d_input_drv, d_output_jit, N,
                                  blockSize, NUM_RUNS);

        // 回读首block结果
        float gpuResult;
        CU_CHECK(cuMemcpyDtoH(&gpuResult, d_output_jit, sizeof(float)));
        float cpuRef = cpuReduce(h_input.data(), blockSize, ops[i]);

        printf("  %-14s | 首block结果: %10.4f (CPU: %10.4f) | 耗时: %.4f ms\n",
               opNames[i], gpuResult, cpuRef, time);
    }

    // ====================================================
    // 实验3: 架构自适应编译 — 展示PTX
    // ====================================================
    printf("\n===================================================\n");
    printf("  实验3: 架构自适应编译\n");
    printf("===================================================\n");
    printf("运行时检测GPU计算能力: sm_%d%d\n", smMajor, smMinor);
    printf("NVRTC编译选项: --gpu-architecture=compute_%d%d\n\n", smMajor,
           smMinor);
    printf("→ 同一程序在不同GPU上会编译出不同架构的PTX:\n");
    printf("  H100 → .target sm_90, A100 → .target sm_80, etc.\n\n");

    // 展示PTX内容（前600字符）
    printf("--- 生成的PTX (前600字符) ---\n");
    int printLen = (int)ptxSum.size() < 600 ? (int)ptxSum.size() : 600;
    printf("%.*s\n...(省略剩余 %zu 字节)...\n", printLen, ptxSum.data(),
           ptxSum.size() - printLen);

    // ---------------------------------------------------
    // 清理资源
    // ---------------------------------------------------
    CU_CHECK(cuMemFree(d_input_drv));
    CU_CHECK(cuMemFree(d_output_jit));
    CUDA_CHECK(cudaFree(d_input_rt));
    CUDA_CHECK(cudaFree(d_output_aot));
    CU_CHECK(cuCtxDestroy(context));

    printf("\n========================================\n");
    printf("  JIT特性演示完成!\n");
    printf("========================================\n");
    printf("\n总结 — JIT编译的核心优势:\n");
    printf("  1. 运行时特化: 运行时值→编译时常量，编译器充分优化\n");
    printf("  2. 动态代码生成: 运行时拼接源码，一套模板生成多种kernel\n");
    printf("  3. 架构自适应: 自动适配当前GPU，无需预编译多个版本\n");

    return 0;
}
