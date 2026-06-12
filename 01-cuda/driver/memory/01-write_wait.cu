/**
 * 例1: cuStreamWriteValue32 + cuStreamWaitValue32 (EQ)
 *
 * 生产者 streamA 填充 buffer → 写 signal=1
 * 消费者 streamB 等 signal==1 → 验证 buffer
 *
 * 编译: make 01
 * 运行: ./01-write_wait
 */

#include <cuda.h>
#include <cstdio>
#include <cstdlib>

#define N (1 << 20)
#define check(e, msg) do { \
    if (e != CUDA_SUCCESS) { fprintf(stderr, "Error %s: %d\n", msg, e); exit(1); } \
} while(0)

// ---- 普通 CUDA kernel (Runtime API 编译), Driver API 只做 stream 同步 ----
__global__ void fill_kernel(float *buf, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) buf[tid] = (float)tid;
}

int main() {
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuDevicePrimaryCtxRetain(&ctx, dev);

    float *d_buf;
    CUdeviceptr buffer, signal;

    // Runtime API 分配 buffer, Driver API 分配 signal
    cudaMalloc(&d_buf, N * sizeof(float));
    buffer = (CUdeviceptr)d_buf;           // Runtime ptr → Driver ptr (直接转)
    cuMemAlloc(&signal, sizeof(int));

    int zero = 0;
    cuMemcpyHtoD(signal, &zero, sizeof(int));

    CUstream streamA, streamB;
    cuStreamCreate(&streamA, 0);
    cuStreamCreate(&streamB, 0);

    // streamA: 跑 kernel → 写 signal = 1
    int block = 256, grid = (N + block - 1) / block;
    fill_kernel<<<grid, block, 0, (cudaStream_t)streamA>>>(d_buf, N);

    check(cuStreamWriteValue32(streamA, signal, 1, CU_STREAM_WRITE_VALUE_DEFAULT),
          "cuStreamWriteValue32");

    // streamB: 等 signal == 1, 然后验证
    check(cuStreamWaitValue32(streamB, signal, 1, CU_STREAM_WAIT_VALUE_EQ),
          "cuStreamWaitValue32");

    float *h_buf = (float*)malloc(N * sizeof(float));
    cudaMemcpyAsync(h_buf, d_buf, N * sizeof(float),
                    cudaMemcpyDeviceToHost, (cudaStream_t)streamB);

    cudaStreamSynchronize((cudaStream_t)streamB);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_buf[i] != (float)i) {
            if (++errors <= 3) printf("  mismatch [%d]: %f\n", i, h_buf[i]);
        }
    }
    printf("  %s streamB waited for signal==1, data verified\n",
           errors == 0 ? "✅" : "❌");

    free(h_buf);
    cudaFree(d_buf); cuMemFree(signal);
    cuStreamDestroy(streamA); cuStreamDestroy(streamB);
    cuDevicePrimaryCtxRelease(dev);
    return errors ? 1 : 0;
}
