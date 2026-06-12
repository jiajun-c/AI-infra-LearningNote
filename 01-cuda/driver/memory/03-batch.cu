/**
 * 例3: cuStreamBatchMemOp — 批量多路同步
 *
 * 4 个生产者并行 kernel + 写各自 flag=1
 * 1 个消费者 batch-wait 所有 4 个 flag (1 次 API 调用)
 * 消费者汇总验证所有 buffer
 *
 * 编译: make 03
 * 运行: ./03-batch
 */

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define K  4
#define N  (1 << 20)
#define check(e, msg) do { \
    if (e != CUDA_SUCCESS) { fprintf(stderr, "Error %s: %d\n", msg, e); exit(1); } \
} while(0)

__global__ void fill_offset(float *buf, int offset, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) buf[tid] = (float)(tid + offset);
}

int main() {
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuDevicePrimaryCtxRetain(&ctx, dev);

    float *d_bufs[K];
    CUdeviceptr flags[K];
    for (int k = 0; k < K; k++) {
        cudaMalloc(&d_bufs[k], N * sizeof(float));
        cuMemAlloc(&flags[k], sizeof(int));
        int z = 0; cuMemcpyHtoD(flags[k], &z, sizeof(int));
    }

    CUstream producers[K], consumer;
    for (int k = 0; k < K; k++)
        cuStreamCreate(&producers[k], 0);
    cuStreamCreate(&consumer, 0);

    int block = 256, grid = (N + block - 1) / block;

    // ---- K 个生产者: kernel → write flag=1 ----
    for (int k = 0; k < K; k++) {
        fill_offset<<<grid, block, 0, (cudaStream_t)producers[k]>>>(
            d_bufs[k], k * 1000, N);
        check(cuStreamWriteValue32(producers[k], flags[k], 1,
                                   CU_STREAM_WRITE_VALUE_DEFAULT),
              "write flag");
    }

    // ---- 消费者: batch-wait 所有 K 个 flag ----
    CUstreamBatchMemOpParams params[K];
    for (int k = 0; k < K; k++) {
        memset(&params[k], 0, sizeof(params[k]));
        params[k].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
        params[k].waitValue.address   = flags[k];
        params[k].waitValue.value     = 1;
        params[k].waitValue.flags     = CU_STREAM_WAIT_VALUE_EQ;
    }
    check(cuStreamBatchMemOp(consumer, K, params, 0), "batchMemOp");

    // 汇总验证
    float *h_buf = (float*)malloc(N * sizeof(float));
    int errors = 0;
    for (int k = 0; k < K; k++) {
        cudaMemcpyAsync(h_buf, d_bufs[k], N * sizeof(float),
                        cudaMemcpyDeviceToHost, (cudaStream_t)consumer);
        cudaStreamSynchronize((cudaStream_t)consumer);

        for (int i = 0; i < N; i++) {
            if (h_buf[i] != (float)(i + k * 1000)) {
                if (++errors <= 3)
                    printf("  buf[%d][%d]: %f != %f\n",
                           k, i, h_buf[i], (float)(i + k * 1000));
            }
        }
    }

    printf("  %s batch-waited %d flags in 1 API call\n",
           errors == 0 ? "✅" : "❌", K);

    free(h_buf);
    for (int k = 0; k < K; k++) {
        cudaFree(d_bufs[k]); cuMemFree(flags[k]);
        cuStreamDestroy(producers[k]);
    }
    cuStreamDestroy(consumer);
    cuDevicePrimaryCtxRelease(dev);
    return errors ? 1 : 0;
}
