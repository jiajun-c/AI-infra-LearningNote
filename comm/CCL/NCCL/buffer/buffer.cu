#include <nccl.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <numeric>
#include <vector>
#include <cmath>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed: NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// =================================================================
// 纯计算密集型 Kernel (Compute Bound)
// 只在寄存器里空转，不读写显存，不抢带宽
// =================================================================
__global__ void compute_bound_kernel(float* dummy_data, int iterations) {
    // 每个线程只读一次，不写回，最大化减少显存压力
    // 这里我们甚至不需要真实的数据依赖，只是为了消耗 SM 时钟周期
    float val = 1.0f;
    if (threadIdx.x == 0) val = dummy_data[0]; 

    #pragma unroll
    for (int k = 0; k < iterations; ++k) {
        // 纯数学运算
        val = sinf(val) * cosf(val) + tanf(val); 
    }
    
    // 防止编译器优化掉循环
    if (val > 10000.0f && threadIdx.x == 0) dummy_data[0] = val;
}

int main() {
    int nGPUs = 0;
    cudaGetDeviceCount(&nGPUs);
    printf("Device Count: %d\n", nGPUs);
    if (nGPUs < 2) { printf("Need at least 2 GPUs\n"); return 0; }

    std::vector<int> devLists(nGPUs);
    std::iota(devLists.begin(), devLists.end(), 0);

    ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * nGPUs);
    NCCLCHECK(ncclCommInitAll(comms, nGPUs, devLists.data()));

    // =============================================================
    // 关键优化 1: 使用小包数据 (Latency Bound)
    // 256KB float = 1MB 数据量
    // 这种大小下，CPU 启动开销(Launch Overhead) 占比很大
    // =============================================================
    size_t count = 256 * 1024; 
    size_t bytes = count * sizeof(float);

    // 统一使用标准 cudaMalloc
    float** sendBuff = (float**)malloc(nGPUs * sizeof(float*));
    float** recvBuff = (float**)malloc(nGPUs * sizeof(float*));
    float** compBuff = (float**)malloc(nGPUs * sizeof(float*)); 

    cudaStream_t* comm_streams = (cudaStream_t*)malloc(nGPUs * sizeof(cudaStream_t));
    cudaStream_t* comp_streams = (cudaStream_t*)malloc(nGPUs * sizeof(cudaStream_t));
    
    // 获取优先级范围 (推荐给 NCCL 高优先级)
    int priority_high, priority_low;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);

    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(devLists[i]));
        CUDACHECK(cudaStreamCreateWithPriority(&comm_streams[i], cudaStreamNonBlocking, priority_high));
        CUDACHECK(cudaStreamCreateWithPriority(&comp_streams[i], cudaStreamNonBlocking, priority_low));

        CUDACHECK(cudaMalloc(&sendBuff[i], bytes));
        CUDACHECK(cudaMalloc(&recvBuff[i], bytes));
        CUDACHECK(cudaMalloc(&compBuff[i], sizeof(float)*1024)); // 计算 buffer 很小即可

        CUDACHECK(cudaMemset(sendBuff[i], 0, bytes));
        CUDACHECK(cudaMemset(recvBuff[i], 0, bytes));
    }

    cudaEvent_t start, stop;
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    // =============================================================
    // 关键优化 2: 增加迭代次数
    // 累积微小的延迟优化，使其变得可测量
    // =============================================================
    const int iter = 2000; 
    
    // 计算 kernel 循环次数：不需要太长，只要稍微有点负载即可
    int kernel_iters = 500; 

    // -------------------------------------------------------------
    // Phase 1: Unregistered (每次通信都有 OS 锁页开销)
    // -------------------------------------------------------------
    printf("\n--- Phase 1: Unregistered (Small Data: %.2f MB, Iters: %d) ---\n", (float)bytes/(1024*1024), iter);
    
    // Warmup
    for(int i=0; i<10; i++){
        NCCLCHECK(ncclGroupStart());
        for(int g=0; g<nGPUs; g++) {
            NCCLCHECK(ncclAllReduce(sendBuff[g], recvBuff[g], count, ncclFloat, ncclSum, comms[g], comm_streams[g]));
        }
        NCCLCHECK(ncclGroupEnd());
        for(int g=0; g<nGPUs; g++) { cudaSetDevice(g); cudaDeviceSynchronize(); }
    }

    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaEventRecord(start, comm_streams[0])); 

    for(int i=0; i<iter; i++){
        // 1. 轻量级计算 (不抢带宽)
        for(int g=0; g<nGPUs; g++) {
            CUDACHECK(cudaSetDevice(g));
            compute_bound_kernel<<<1, 256, 0, comp_streams[g]>>>(compBuff[g], kernel_iters);
        }

        // 2. 通信 (小包)
        NCCLCHECK(ncclGroupStart());
        for(int g=0; g<nGPUs; g++) {
            NCCLCHECK(ncclAllReduce(sendBuff[g], recvBuff[g], count, ncclFloat, ncclSum, comms[g], comm_streams[g]));
        }
        NCCLCHECK(ncclGroupEnd());
    }

    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaEventRecord(stop, comm_streams[0]));
    CUDACHECK(cudaEventSynchronize(stop));

    float ms_unreg = 0;
    CUDACHECK(cudaEventElapsedTime(&ms_unreg, start, stop));
    printf("Unregistered Total Time: %.3f ms\n", ms_unreg);
    printf("Avg Time per Iter:       %.3f us\n", (ms_unreg * 1000) / iter);


    // -------------------------------------------------------------
    // Phase 2: Registered (零拷贝启动，无 OS 开销)
    // -------------------------------------------------------------
    printf("\n--- Phase 2: Registered (Small Data: %.2f MB, Iters: %d) ---\n", (float)bytes/(1024*1024), iter);

    std::vector<void*> send_reg_handles(nGPUs);
    std::vector<void*> recv_reg_handles(nGPUs);

    // 注册内存
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommRegister(comms[i], sendBuff[i], bytes, &send_reg_handles[i]));
        NCCLCHECK(ncclCommRegister(comms[i], recvBuff[i], bytes, &recv_reg_handles[i]));
    }

    // Warmup
    for(int i=0; i<10; i++){
        NCCLCHECK(ncclGroupStart());
        for(int g=0; g<nGPUs; g++) {
            NCCLCHECK(ncclAllReduce(sendBuff[g], recvBuff[g], count, ncclFloat, ncclSum, comms[g], comm_streams[g]));
        }
        NCCLCHECK(ncclGroupEnd());
        for(int g=0; g<nGPUs; g++) { cudaSetDevice(g); cudaDeviceSynchronize(); }
    }

    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaEventRecord(start, comm_streams[0]));

    for(int i=0; i<iter; i++){
        // 1. 轻量级计算
        for(int g=0; g<nGPUs; g++) {
            CUDACHECK(cudaSetDevice(g));
            compute_bound_kernel<<<1, 256, 0, comp_streams[g]>>>(compBuff[g], kernel_iters);
        }

        // 2. 通信 (已注册)
        NCCLCHECK(ncclGroupStart());
        for(int g=0; g<nGPUs; g++) {
            NCCLCHECK(ncclAllReduce(sendBuff[g], recvBuff[g], count, ncclFloat, ncclSum, comms[g], comm_streams[g]));
        }
        NCCLCHECK(ncclGroupEnd());
    }

    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaEventRecord(stop, comm_streams[0]));
    CUDACHECK(cudaEventSynchronize(stop));

    float ms_reg = 0;
    CUDACHECK(cudaEventElapsedTime(&ms_reg, start, stop));
    printf("Registered   Total Time: %.3f ms\n", ms_reg);
    printf("Avg Time per Iter:       %.3f us\n", (ms_reg * 1000) / iter);

    // -------------------------------------------------------------
    // 结果对比
    // -------------------------------------------------------------
    printf("\n------------------------------------------------\n");
    float improvement = (ms_unreg - ms_reg) / ms_unreg * 100.0f;
    printf("Latency Improvement: %.2f%%\n", improvement);
    printf("Speedup Factor:      %.2fx\n", ms_unreg / ms_reg);
    printf("------------------------------------------------\n");

    // 清理
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommDeregister(comms[i], send_reg_handles[i]));
        NCCLCHECK(ncclCommDeregister(comms[i], recv_reg_handles[i]));
        cudaFree(sendBuff[i]);
        cudaFree(recvBuff[i]);
        cudaFree(compBuff[i]);
        cudaStreamDestroy(comm_streams[i]);
        cudaStreamDestroy(comp_streams[i]);
        ncclCommDestroy(comms[i]);
    }
    free(comms);
    free(sendBuff); free(recvBuff); free(compBuff);

    return 0;
}