#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <cuda_runtime.h>

// 宏定义用于检查 CUDA API 错误
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        std::cout << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// ================= Device Kernel Code (包含修正) =================

__device__ float warpReduceMax(float var) {
    for (int i = 16; i > 0; i >>= 1) {
        var = max(var, __shfl_down_sync(0xffffffff, var, i));
    }
    return var;
}

__device__ float warpReduceSum(float var) {
    for (int i = 16; i > 0; i >>= 1) {
        var += __shfl_down_sync(0xffffffff, var, i);
    }
    return var;
}

__global__ void softmax_kernel(const float *input, float *output, int seq_len) {
    // 假设 GridDim.x = Batch Size, 一个 Block 处理一行
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int base = bid * seq_len;
    
    int warpId = tid / 32;
    int laneId = tid % 32;
    // 计算当前 Block 有多少个 Warp (例如 256线程 -> 8 Warps)
    int warpCount = blockDim.x / 32; 

    __shared__ float s_val[32]; // 只有 32 个槽位，对应 32 个 Warp

    // ================= Step 1: Max Reduction =================
    float local_max = -FLT_MAX;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        local_max = max(local_max, input[base + i]);
    }
    
    local_max = warpReduceMax(local_max);
    
    // 【修正】这里必须用 warpId，不能用 tid
    if (laneId == 0) {
        if(warpId < 32) s_val[warpId] = local_max; 
    }
    __syncthreads();

    if (warpId == 0) {
        local_max = (laneId < warpCount) ? s_val[laneId] : -FLT_MAX;
        local_max = warpReduceMax(local_max);
    }
    // 广播给所有线程
    if (tid == 0) s_val[0] = local_max;
    __syncthreads();
    
    float blockMax = s_val[0];

    // ================= Step 2: Sum Reduction =================
    float sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float x = input[base + i] - blockMax;
        float ex = expf(x);
        sum += ex;
    }
    
    sum = warpReduceSum(sum);
    
    // 【修正】这里必须用 warpId，你原来的代码写的是 s_val[tid]，会导致越界
    if (laneId == 0) {
        if(warpId < 32) s_val[warpId] = sum;
    }
    __syncthreads();

    if (warpId == 0) {
        sum = (laneId < warpCount) ? s_val[laneId] : 0.0f;
        sum = warpReduceSum(sum);
    }
    // 广播给所有线程
    if (tid == 0) s_val[0] = sum;
    __syncthreads();
    
    float wholeSum = s_val[0];

    // ================= Step 3: Write Output =================
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float x = input[base + i] - blockMax;
        float ex = expf(x);
        output[base + i] = ex / wholeSum;
    }
}

// ================= Host Code (CPU Reference & Main) =================

// CPU 版本的 Softmax 用于验证结果正确性
void cpu_softmax(const std::vector<float>& h_input, std::vector<float>& h_ref, int batch_size, int seq_len) {
    for (int b = 0; b < batch_size; ++b) {
        int base = b * seq_len;
        
        // 1. Find Max
        float max_val = -FLT_MAX;
        for (int i = 0; i < seq_len; ++i) {
            max_val = std::max(max_val, h_input[base + i]);
        }
        
        // 2. Compute Sum of Exp
        float sum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            sum += std::exp(h_input[base + i] - max_val);
        }
        
        // 3. Normalize
        for (int i = 0; i < seq_len; ++i) {
            h_ref[base + i] = std::exp(h_input[base + i] - max_val) / sum;
        }
    }
}

int main() {
    // 1. 设置测试参数
    const int batch_size = 10;
    const int seq_len = 2048; // 设置一个较长的序列以测试循环逻辑
    size_t num_elements = batch_size * seq_len;
    size_t size_bytes = num_elements * sizeof(float);

    std::cout << "Testing Softmax Kernel..." << std::endl;
    std::cout << "Batch Size: " << batch_size << ", Seq Len: " << seq_len << std::endl;

    // 2. 分配主机内存
    std::vector<float> h_input(num_elements);
    std::vector<float> h_output(num_elements);
    std::vector<float> h_ref(num_elements);

    // 3. 初始化输入数据 (随机数)
    srand(time(NULL));
    for (size_t i = 0; i < num_elements; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX; // 0.0 ~ 1.0
    }

    // 4. 计算 CPU 参考结果
    cpu_softmax(h_input, h_ref, batch_size, seq_len);

    // 5. 分配设备内存
    float *d_input, *d_output;
    CHECK(cudaMalloc(&d_input, size_bytes));
    CHECK(cudaMalloc(&d_output, size_bytes));

    // 6. 拷贝数据到设备
    CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));

    // 7. 启动 Kernel
    // Block 大小设为 256 (8 Warps)，足以测试 Warp 间规约逻辑
    int blockSize = 256; 
    // Grid 大小设为 Batch Size (一行一个 Block)
    int gridSize = batch_size;

    softmax_kernel<<<gridSize, blockSize>>>(d_input, d_output, seq_len);
    CHECK(cudaGetLastError()); // 检查启动错误
    CHECK(cudaDeviceSynchronize());

    // 8. 拷贝结果回主机
    CHECK(cudaMemcpy(h_output.data(), d_output, size_bytes, cudaMemcpyDeviceToHost));

    // 9. 验证结果
    bool passed = true;
    double max_diff = 0.0;
    for (size_t i = 0; i < num_elements; ++i) {
        double diff = std::abs(h_output[i] - h_ref[i]);
        if (diff > 1e-5) { // 浮点误差容忍度
            std::cout << "Mismatch at index " << i << ": CPU=" << h_ref[i] 
                      << ", GPU=" << h_output[i] << ", Diff=" << diff << std::endl;
            passed = false;
            break; 
        }
        if (diff > max_diff) max_diff = diff;
    }

    if (passed) {
        std::cout << "TEST PASSED!" << std::endl;
        std::cout << "Max Error: " << max_diff << std::endl;
    } else {
        std::cout << "TEST FAILED!" << std::endl;
    }

    // 10. 清理内存
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}