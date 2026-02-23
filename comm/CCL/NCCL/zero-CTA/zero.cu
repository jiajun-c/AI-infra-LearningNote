#include <nccl.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <numeric>
#include <vector>
#include <omp.h>
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

// 纯计算 Kernel (模拟计算负载)
__global__ void compute_bound_kernel(float* dummy_data, int iterations) {
    float val = 1.0f;
    if (threadIdx.x == 0) val = dummy_data[0]; 
    #pragma unroll
    for (int k = 0; k < iterations; ++k) {
        val = sinf(val) * cosf(val) + tanf(val); 
    }
    if (val > 10000.0f && threadIdx.x == 0) dummy_data[0] = val;
}

int main() {
    int nGPUs = 0;
    cudaGetDeviceCount(&nGPUs);
    printf("Device Count: %d\n", nGPUs);
    if (nGPUs < 2) { printf("Need at least 2 GPUs\n"); return 0; }

    std::vector<int> devLists(nGPUs);
    std::iota(devLists.begin(), devLists.end(), 0);

    // =============================================================
    // 通用资源分配
    // =============================================================
    size_t count = 256 * 1024; // 1MB 数据 (Latency Sensitive)
    size_t bytes = count * sizeof(float);

    // Host 指针数组，用于存储 Device 指针
    float** sendBuff = (float**)malloc(nGPUs * sizeof(float*));
    float** recvBuff = (float**)malloc(nGPUs * sizeof(float*));
    float** compBuff = (float**)malloc(nGPUs * sizeof(float*)); 
    
    ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * nGPUs);
    cudaStream_t* comm_streams = (cudaStream_t*)malloc(nGPUs * sizeof(cudaStream_t));
    cudaStream_t* comp_streams = (cudaStream_t*)malloc(nGPUs * sizeof(cudaStream_t));
    
    int priority_high, priority_low;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);

    // 初始化流和计算 Buffer (计算 Buffer 整个过程复用)
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(devLists[i]));
        CUDACHECK(cudaStreamCreateWithPriority(&comm_streams[i], cudaStreamNonBlocking, priority_high));
        CUDACHECK(cudaStreamCreateWithPriority(&comp_streams[i], cudaStreamNonBlocking, priority_low));
        CUDACHECK(cudaMalloc(&compBuff[i], sizeof(float)*1024));
    }

    // 初始分配普通 CUDA 内存 (Phase 1-3)
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(devLists[i]));
        CUDACHECK(cudaMalloc(&sendBuff[i], bytes));
        CUDACHECK(cudaMalloc(&recvBuff[i], bytes));
        CUDACHECK(cudaMemset(sendBuff[i], 0, bytes));
        CUDACHECK(cudaMemset(recvBuff[i], 0, bytes));
    }

    cudaEvent_t start, stop;
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    const int iter = 2000; 
    int kernel_iters = 500; 

    // 初始化默认 Comm (Phase 1 & 2 使用)
    printf("Initializing Standard NCCL Comm...\n");
    NCCLCHECK(ncclCommInitAll(comms, nGPUs, devLists.data()));

    // =============================================================
    // Phase 1: Unregistered (Standard)
    // =============================================================
    printf("\n--- Phase 1: Unregistered (Standard CTA) ---\n");
    
    // Warmup
    for(int i=0; i<5; i++){
        NCCLCHECK(ncclGroupStart());
        for(int g=0; g<nGPUs; g++) NCCLCHECK(ncclAllReduce(sendBuff[g], recvBuff[g], count, ncclFloat, ncclSum, comms[g], comm_streams[g]));
        NCCLCHECK(ncclGroupEnd());
        for(int g=0; g<nGPUs; g++) { cudaSetDevice(g); cudaDeviceSynchronize(); }
    }

    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaEventRecord(start, comm_streams[0])); 

    for(int i=0; i<iter; i++){
        #pragma omp parallel for num_threads(nGPUs)
        for(int g=0; g<nGPUs; g++) {
            CUDACHECK(cudaSetDevice(g));
            compute_bound_kernel<<<1, 256, 0, comp_streams[g]>>>(compBuff[g], kernel_iters);
        }

        NCCLCHECK(ncclGroupStart());
        for(int g=0; g<nGPUs; g++) NCCLCHECK(ncclAllReduce(sendBuff[g], recvBuff[g], count, ncclFloat, ncclSum, comms[g], comm_streams[g]));
        NCCLCHECK(ncclGroupEnd());
    }

    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaEventRecord(stop, comm_streams[0]));
    CUDACHECK(cudaEventSynchronize(stop));

    float ms_unreg = 0;
    CUDACHECK(cudaEventElapsedTime(&ms_unreg, start, stop));
    printf("Time: %.3f ms | Avg: %.3f us\n", ms_unreg, (ms_unreg * 1000) / iter);


    // =============================================================
    // Phase 2: Registered (Standard)
    // =============================================================
    printf("\n--- Phase 2: Registered (Standard CTA) ---\n");

    std::vector<void*> send_handles(nGPUs), recv_handles(nGPUs);
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommRegister(comms[i], sendBuff[i], bytes, &send_handles[i]));
        NCCLCHECK(ncclCommRegister(comms[i], recvBuff[i], bytes, &recv_handles[i]));
    }

    // Warmup
    for(int i=0; i<5; i++){
        NCCLCHECK(ncclGroupStart());
        for(int g=0; g<nGPUs; g++) NCCLCHECK(ncclAllReduce(sendBuff[g], recvBuff[g], count, ncclFloat, ncclSum, comms[g], comm_streams[g]));
        NCCLCHECK(ncclGroupEnd());
        for(int g=0; g<nGPUs; g++) { cudaSetDevice(g); cudaDeviceSynchronize(); }
    }

    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaEventRecord(start, comm_streams[0]));

    for(int i=0; i<iter; i++){
        #pragma omp parallel for num_threads(nGPUs)
        for(int g=0; g<nGPUs; g++) {
            CUDACHECK(cudaSetDevice(g));
            compute_bound_kernel<<<1, 256, 0, comp_streams[g]>>>(compBuff[g], kernel_iters);
        }
        
        NCCLCHECK(ncclGroupStart());
        for(int g=0; g<nGPUs; g++) NCCLCHECK(ncclAllReduce(sendBuff[g], recvBuff[g], count, ncclFloat, ncclSum, comms[g], comm_streams[g]));
        NCCLCHECK(ncclGroupEnd());
    }

    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaEventRecord(stop, comm_streams[0]));
    CUDACHECK(cudaEventSynchronize(stop));

    float ms_reg = 0;
    CUDACHECK(cudaEventElapsedTime(&ms_reg, start, stop));
    printf("Time: %.3f ms | Avg: %.3f us\n", ms_reg, (ms_reg * 1000) / iter);

    // 清理 Phase 2 注册，准备销毁 comms
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommDeregister(comms[i], send_handles[i]));
        NCCLCHECK(ncclCommDeregister(comms[i], recv_handles[i]));
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }


    // =============================================================
    // Phase 3: Registered (Min-Channels)
    // =============================================================
    printf("\n--- Phase 3: Registered (Min-CTA / 1 Channel) ---\n");
    
    // 设置环境变量
    setenv("NCCL_MIN_NCHANNELS", "1", 1);
    setenv("NCCL_MAX_NCHANNELS", "1", 1);
    
    printf("Re-initializing NCCL with 1 Channel...\n");
    NCCLCHECK(ncclCommInitAll(comms, nGPUs, devLists.data()));

    // 注册 (Comm 是新的，需要重新注册)
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommRegister(comms[i], sendBuff[i], bytes, &send_handles[i]));
        NCCLCHECK(ncclCommRegister(comms[i], recvBuff[i], bytes, &recv_handles[i]));
    }

    // Warmup
    for(int i=0; i<5; i++){
        NCCLCHECK(ncclGroupStart());
        for(int g=0; g<nGPUs; g++) NCCLCHECK(ncclAllReduce(sendBuff[g], recvBuff[g], count, ncclFloat, ncclSum, comms[g], comm_streams[g]));
        NCCLCHECK(ncclGroupEnd());
        for(int g=0; g<nGPUs; g++) { cudaSetDevice(g); cudaDeviceSynchronize(); }
    }

    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaEventRecord(start, comm_streams[0]));

    for(int i=0; i<iter; i++){
        #pragma omp parallel for num_threads(nGPUs)
        for(int g=0; g<nGPUs; g++) {
            CUDACHECK(cudaSetDevice(g));
            compute_bound_kernel<<<1, 256, 0, comp_streams[g]>>>(compBuff[g], kernel_iters);
        }

        NCCLCHECK(ncclGroupStart());
        for(int g=0; g<nGPUs; g++) NCCLCHECK(ncclAllReduce(sendBuff[g], recvBuff[g], count, ncclFloat, ncclSum, comms[g], comm_streams[g]));
        NCCLCHECK(ncclGroupEnd());
    }

    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaEventRecord(stop, comm_streams[0]));
    CUDACHECK(cudaEventSynchronize(stop));

    float ms_min_cta = 0;
    CUDACHECK(cudaEventElapsedTime(&ms_min_cta, start, stop));
    printf("Time: %.3f ms | Avg: %.3f us\n", ms_min_cta, (ms_min_cta * 1000) / iter);

    // === Phase 3 彻底清理 (释放 cudaMalloc 内存) ===
    // 必须释放掉，因为 Phase 4 将使用 ncclMemAlloc
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommDeregister(comms[i], send_handles[i]));
        NCCLCHECK(ncclCommDeregister(comms[i], recv_handles[i]));
        NCCLCHECK(ncclCommDestroy(comms[i])); 
        CUDACHECK(cudaFree(sendBuff[i])); // 释放普通内存
        CUDACHECK(cudaFree(recvBuff[i]));
    }


    // =============================================================
    // Phase 4: Zero-CTA (Integrated)
    // =============================================================
    printf("\n--- Phase 4: Zero-CTA (Alloc + WinRegister) ---\n");

    // 1. 清除 Phase 3 的环境变量，防止影响
    unsetenv("NCCL_MIN_NCHANNELS");
    unsetenv("NCCL_MAX_NCHANNELS");

    // 2. 配置 Zero-CTA
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.CTAPolicy = NCCL_CTA_POLICY_ZERO; 

    ncclUniqueId id;
    NCCLCHECK(ncclGetUniqueId(&id));

    // 3. 重新初始化 Comms (带 Config)
    printf("Re-initializing NCCL for Zero-CTA...\n");
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < nGPUs; i++) {
        cudaSetDevice(devLists[i]);
        NCCLCHECK(ncclCommInitRankConfig(&comms[i], nGPUs, id, i, &config));
    }
    NCCLCHECK(ncclGroupEnd());

    // 4. 使用 ncclMemAlloc 分配内存
    // 注意：sendBuff/recvBuff 已经在 Phase 3 结束时被 cudaFree 释放，这里复用指针变量
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPUs; i++) {
        cudaSetDevice(devLists[i]);
        NCCLCHECK(ncclMemAlloc((void**)&sendBuff[i], bytes));
        NCCLCHECK(ncclMemAlloc((void**)&recvBuff[i], bytes));
        CUDACHECK(cudaMemset(sendBuff[i], 0, bytes)); // 也可以用 cudaMemset
        CUDACHECK(cudaMemset(recvBuff[i], 0, bytes));
    }
    NCCLCHECK(ncclGroupEnd());

    // 5. 注册 Windows
    std::vector<ncclWindow_t> src_wins(nGPUs);
    std::vector<ncclWindow_t> dst_wins(nGPUs);

    printf("Registering Windows...\n");
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPUs; i++) {
        cudaSetDevice(devLists[i]);
        NCCLCHECK(ncclCommWindowRegister(comms[i], sendBuff[i], bytes, &src_wins[i], NCCL_WIN_COLL_SYMMETRIC));
        NCCLCHECK(ncclCommWindowRegister(comms[i], recvBuff[i], bytes, &dst_wins[i], NCCL_WIN_COLL_SYMMETRIC));
    }
    NCCLCHECK(ncclGroupEnd());

    // Warmup
    for(int i=0; i<5; i++){
        NCCLCHECK(ncclGroupStart());
        for(int g=0; g<nGPUs; g++) NCCLCHECK(ncclAllReduce(sendBuff[g], recvBuff[g], count, ncclFloat, ncclSum, comms[g], comm_streams[g]));
        NCCLCHECK(ncclGroupEnd());
        for(int g=0; g<nGPUs; g++) { cudaSetDevice(g); cudaDeviceSynchronize(); }
    }

    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaEventRecord(start, comm_streams[0]));

    for(int i=0; i<iter; i++){
        #pragma omp parallel for num_threads(nGPUs)
        for(int g=0; g<nGPUs; g++) {
            CUDACHECK(cudaSetDevice(g));
            compute_bound_kernel<<<1, 256, 0, comp_streams[g]>>>(compBuff[g], kernel_iters);
        }

        NCCLCHECK(ncclGroupStart());
        for(int g=0; g<nGPUs; g++) NCCLCHECK(ncclAllReduce(sendBuff[g], recvBuff[g], count, ncclFloat, ncclSum, comms[g], comm_streams[g]));
        NCCLCHECK(ncclGroupEnd());
    }

    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaEventRecord(stop, comm_streams[0]));
    CUDACHECK(cudaEventSynchronize(stop));

    float ms_zero_cta = 0;
    CUDACHECK(cudaEventElapsedTime(&ms_zero_cta, start, stop));
    printf("Time: %.3f ms | Avg: %.3f us\n", ms_zero_cta, (ms_zero_cta * 1000) / iter);


    // =============================================================
    // 总结对比
    // =============================================================
    printf("\n------------------------------------------------\n");
    printf("Baseline (Unreg):   %.3f us\n", (ms_unreg * 1000) / iter);
    printf("Registered:         %.3f us (Speedup: %.2fx)\n", (ms_reg * 1000) / iter, ms_unreg/ms_reg);
    printf("Reg + Min-CTA:      %.3f us (Speedup: %.2fx)\n", (ms_min_cta * 1000) / iter, ms_unreg/ms_min_cta);
    printf("Zero-CTA:           %.3f us (Speedup: %.2fx)\n", (ms_zero_cta * 1000) / iter, ms_unreg/ms_zero_cta);
    printf("------------------------------------------------\n");

    // Phase 4 专用清理 (使用 ncclMemFree 等)
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        // Deregister Windows
        NCCLCHECK(ncclCommWindowDeregister(comms[i], src_wins[i]));
        NCCLCHECK(ncclCommWindowDeregister(comms[i], dst_wins[i]));
        
        // Free NCCL Allocated Memory
        NCCLCHECK(ncclMemFree(sendBuff[i]));
        NCCLCHECK(ncclMemFree(recvBuff[i]));
        
        // Destroy Comm
        NCCLCHECK(ncclCommDestroy(comms[i]));
        
        // Free 其他 CUDA 资源
        cudaStreamDestroy(comm_streams[i]);
        cudaStreamDestroy(comp_streams[i]);
        cudaFree(compBuff[i]);
    }

    // 释放 Host 数组
    free(comms); free(sendBuff); free(recvBuff); free(compBuff);
    free(comm_streams); free(comp_streams);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}