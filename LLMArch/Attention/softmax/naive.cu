#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>
__global__ void safeSoftMax(float *input, float *output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > rows ) return;
    float max_val = -INFINITY;
    for (int i = 0; i < cols; i++) {
        float val = input[row * cols + i];
        if (val > max_val) max_val = val;
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < cols; i++) {
        float exp_val = expf(input[row * cols + i] - max_val);
        sum_exp += exp_val;
        output[row * cols + i] = exp_val;
    }
    for (int i = 0; i < cols; i++) {
        output[row * cols + i] /= sum_exp;
    }
}   

__global__ void onlineSoftMax(float *input, float *output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > rows ) return;
    float max_val = 0.0f;
    float now_val = 0.0f;
    float sum_exp = 0.0f;
    for (int i = 0; i < cols; i++) {
        now_val = max(max_val, input[row * cols + i]);
        float exp_val = expf(input[row * cols + i] - max_val);
        sum_exp = sum_exp * (now_val - max_val)  + exp_val;
        max_val = now_val;
    }
    for (int i = 0; i < cols; i++) {
        output[row * cols + i] = expf(input[row * cols + i] - max_val) / sum_exp;
    }
}   

void launchsafeSoftMax(float *input, float *output, int rows, int cols) {
    dim3 blockSize(1024);
    dim3 gridSize((rows + blockSize.y - 1)/blockSize.x);
    safeSoftMax<<<gridSize, blockSize>>>(input, output, rows, cols);
}

int main() {
    const int rows = 4096*32;
    const int cols = 16;

    // 分配主机内存
    float* h_input = new float[rows * cols];
    float* h_output = new float[rows * cols];
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 初始化输入数据（示例）
    for (int i = 0; i < rows * cols; ++i) {
        h_input[i] = static_cast<float>(i % 10 + 1);
    }

    // 分配设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));

    // 数据传输
    cudaMemcpy(d_input, h_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    // 执行Kernel
    launchsafeSoftMax(d_input, d_output, rows, cols);
    cudaEventRecord(stop);
    // 取回结果（示例检查第一行第一个元素）
    cudaMemcpy(h_output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    // for (int i = 0; i < 10; i++)
    // printf("Output[0][0] = %f\n", h_output[i]);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds);
    // 释放内存
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
    
    return 0;
}