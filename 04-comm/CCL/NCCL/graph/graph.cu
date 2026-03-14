#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <nccl.h>

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

// ----------------------------------------------------------------
// Kernel A: 模拟计算梯度 (例如: x = x + 1)
// ----------------------------------------------------------------
__global__ void kernel_A(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] + 1.0f; 
    }
}

// ----------------------------------------------------------------
// Kernel C: 模拟参数更新 (例如: x = x * 0.5)
// ----------------------------------------------------------------
__global__ void kernel_C(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * 0.5f; 
    }
}

int main(int argc, char* argv[]) {
    // 1. 环境检查
    int nGPUs = 0;
    cudaGetDeviceCount(&nGPUs);
    if (nGPUs < 2) {
        printf("Error: This test requires at least 2 GPUs.\n");
        return 0;
    }
    printf("Running Multi-Stage Graph Capture on %d GPUs...\n", nGPUs);

    // 2. 初始化 NCCL
    std::vector<int> devs(nGPUs);
    std::iota(devs.begin(), devs.end(), 0);
    ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * nGPUs);
    NCCLCHECK(ncclCommInitAll(comms, nGPUs, devs.data()));

    // 3. 资源分配
    int size = 1024 * 1024; 
    size_t bytes = size * sizeof(float);
    
    std::vector<float*> sendbuff(nGPUs);
    std::vector<float*> recvbuff(nGPUs);
    std::vector<cudaStream_t> streams(nGPUs);
    std::vector<cudaGraph_t> graphs(nGPUs);
    std::vector<cudaGraphExec_t> graphExecs(nGPUs);

    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        // 使用 NonBlocking 流以避免 NVLS 冲突
        CUDACHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        CUDACHECK(cudaMalloc(&sendbuff[i], bytes));
        CUDACHECK(cudaMalloc(&recvbuff[i], bytes));
        CUDACHECK(cudaMemset(sendbuff[i], 0, bytes));
        CUDACHECK(cudaMemset(recvbuff[i], 0, bytes));
    }

    // =================================================================
    // 4. Warmup (关键！防止 Graph Capture 报错)
    // =================================================================
    printf("Warming up to init NVLS/Buffers...\n");
    for(int i=0; i<3; i++) {
        for(int g=0; g<nGPUs; g++) {
            CUDACHECK(cudaSetDevice(g));
            kernel_A<<< (size+255)/256, 256, 0, streams[g] >>>(sendbuff[g], size);
        }
        NCCLCHECK(ncclGroupStart());
        for(int g=0; g<nGPUs; g++) {
            NCCLCHECK(ncclAllReduce(sendbuff[g], recvbuff[g], size, ncclFloat, ncclSum, comms[g], streams[g]));
        }
        NCCLCHECK(ncclGroupEnd());
        for(int g=0; g<nGPUs; g++) {
            CUDACHECK(cudaSetDevice(g));
            kernel_C<<< (size+255)/256, 256, 0, streams[g] >>>(recvbuff[g], size);
        }
    }
    for(int g=0; g<nGPUs; g++) {
        CUDACHECK(cudaSetDevice(g));
        CUDACHECK(cudaDeviceSynchronize()); // 确保预热彻底完成
    }

    // =================================================================
    // 5. 核心：构建流水线图 (Capture)
    // 逻辑：Begin -> Kernel_A -> AllReduce -> Kernel_C -> End
    // =================================================================
    printf("Capturing Graphs...\n");

    // A. 所有 Rank 开始捕获
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamBeginCapture(streams[i], cudaStreamCaptureModeGlobal));
    }

    // B. 所有 Rank 录制 Kernel A
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        kernel_A<<< (size+255)/256, 256, 0, streams[i] >>>(sendbuff[i], size);
    }

    // C. 所有 Rank 录制 NCCL AllReduce (必须用 Group 包裹)
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        // 注意：实际场景中通常是对 A 的输出做 Reduce，这里简化为对 sendbuff
        NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, 
                                ncclFloat, ncclSum, comms[i], streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());

    // D. 所有 Rank 录制 Kernel C
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        // Kernel C 通常处理 Reduce 的结果
        kernel_C<<< (size+255)/256, 256, 0, streams[i] >>>(recvbuff[i], size);
    }

    // E. 所有 Rank 结束捕获
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamEndCapture(streams[i], &graphs[i]));
    }

    // =================================================================
    // 6. 实例化 (Instantiate)
    // =================================================================
    printf("Instantiating...\n");
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaGraphInstantiate(&graphExecs[i], graphs[i], NULL, NULL, 0));
    }

    // =================================================================
    // 7. 执行 (Launch)
    // =================================================================
    printf("Launching Graph 10 times...\n");
    
    // 逻辑流：
    // Iter 1: Send=0 -> KernelA(+1) -> Send=1 -> AllReduce(Sum) -> Recv=N -> KernelC(*0.5) -> Recv=N/2
    // Iter 2: Send=1 -> KernelA(+1) -> Send=2 -> AllReduce(Sum) -> Recv=2N -> KernelC(*0.5) -> Recv=N
    // 注意：因为我这里 Kernel_A 改写的是 sendbuff，Kernel_C 改写的是 recvbuff。
    // 这只是个演示逻辑。
    
    for (int step = 0; step < 10; step++) {
        for (int i = 0; i < nGPUs; i++) {
            CUDACHECK(cudaSetDevice(i));
            CUDACHECK(cudaGraphLaunch(graphExecs[i], streams[i]));
        }
    }

    // =================================================================
    // 8. 验证与清理
    // =================================================================
    printf("Synchronizing...\n");
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
        
        // 简单清理
        CUDACHECK(cudaGraphExecDestroy(graphExecs[i]));
        CUDACHECK(cudaGraphDestroy(graphs[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
        ncclCommDestroy(comms[i]);
    }
    free(comms);

    printf("Test Finished Successfully.\n");
    return 0;
}