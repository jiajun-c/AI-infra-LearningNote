/**
 * 例2: 多阶段流水线 — GEQ 条件
 *
 * stage 0: init buffer[i]=i        → signal=1
 * stage 1: wait signal>=1, buf*=2  → signal=2
 * stage 2: wait signal>=2, buf*=0.5 → signal=3
 *
 * 最终: buf[i] = i*2*0.5 = i (回到原值, 验证正确性)
 *
 * 编译: make 02
 * 运行: ./02-pipeline
 */

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define N (1 << 20)
#define check(e, msg) do { \
    if (e != CUDA_SUCCESS) { fprintf(stderr, "Error %s: %d\n", msg, e); exit(1); } \
} while(0)

__global__ void scale_kernel(float *buf, float factor, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) buf[tid] *= factor;
}

int main() {
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuDevicePrimaryCtxRetain(&ctx, dev);

    float *d_buf;
    CUdeviceptr signal;
    cudaMalloc(&d_buf, N * sizeof(float));
    cuMemAlloc(&signal, sizeof(int));
    int zero = 0; cuMemcpyHtoD(signal, &zero, sizeof(int));

    CUstream s0, s1, s2;
    cuStreamCreate(&s0, 0); cuStreamCreate(&s1, 0); cuStreamCreate(&s2, 0);

    int block = 256, grid = (N + block - 1) / block;

    // ---- stage 0: 初始化 → signal=1 ----
    float *h_init = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_init[i] = (float)i;
    cudaMemcpyAsync(d_buf, h_init, N * sizeof(float),
                    cudaMemcpyHostToDevice, (cudaStream_t)s0);
    check(cuStreamWriteValue32(s0, signal, 1, CU_STREAM_WRITE_VALUE_DEFAULT),
          "write sig=1");
    printf("  stage 0: init buffer, signal=1\n");

    // ---- stage 1: wait GEQ 1 → buf*=2.0 → signal=2 ----
    check(cuStreamWaitValue32(s1, signal, 1, CU_STREAM_WAIT_VALUE_GEQ),
          "wait GEQ 1");
    scale_kernel<<<grid, block, 0, (cudaStream_t)s1>>>(d_buf, 2.0f, N);
    check(cuStreamWriteValue32(s1, signal, 2, CU_STREAM_WRITE_VALUE_DEFAULT),
          "write sig=2");
    printf("  stage 1: wait GEQ 1 → *2.0 → signal=2\n");

    // ---- stage 2: wait GEQ 2 → buf*=0.5 → signal=3 ----
    check(cuStreamWaitValue32(s2, signal, 2, CU_STREAM_WAIT_VALUE_GEQ),
          "wait GEQ 2");
    scale_kernel<<<grid, block, 0, (cudaStream_t)s2>>>(d_buf, 0.5f, N);
    check(cuStreamWriteValue32(s2, signal, 3, CU_STREAM_WRITE_VALUE_DEFAULT),
          "write sig=3");
    printf("  stage 2: wait GEQ 2 → *0.5 → signal=3\n");

    cudaStreamSynchronize((cudaStream_t)s2);

    float *h_buf = (float*)malloc(N * sizeof(float));
    cudaMemcpy(h_buf, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_buf[i] - (float)i) > 0.01f) {
            if (++errors <= 3) printf("  mismatch [%d]: %f != %f\n", i, h_buf[i], (float)i);
        }
    }

    int h_sig; cuMemcpyDtoH(&h_sig, signal, sizeof(int));
    printf("  final signal=%d, %s pipeline verified\n",
           h_sig, errors == 0 ? "✅" : "❌");

    free(h_init); free(h_buf);
    cudaFree(d_buf); cuMemFree(signal);
    cuStreamDestroy(s0); cuStreamDestroy(s1); cuStreamDestroy(s2);
    cuDevicePrimaryCtxRelease(dev);
    return errors ? 1 : 0;
}
