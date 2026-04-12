#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>

// ============================================================================
// Device Functions
// ============================================================================

template<typename T, int K>
__device__ __forceinline__ void insert_to_local_rank(
    T* local_vals,
    int* local_ids,
    T val,
    int id
) {
    // 如果比最小值还要小，直接跳过
    if (val <= local_vals[K - 1]) return;
    local_vals[K - 1] = val;
    local_ids[K - 1]  = id;

    #pragma unroll
    for (int i = K - 2; i >= 0; --i)
    {
        if (local_vals[i] < local_vals[i + 1]) {
            T tmp_v = local_vals[i];
            local_vals[i] = local_vals[i+1];
            local_vals[i+1] = tmp_v;

            int tmp_id = local_ids[i];
            local_ids[i] = local_ids[i+1];
            local_ids[i+1] = tmp_id;
        } else {
            break;
        }
    }
}

// ============================================================================
// TopK Kernel - Small K version (K is compile-time constant)
// ============================================================================

template<typename T, int K>
__global__ void topk_kernel_small_k(
    const T* __restrict__ input,      // shape: [batch_size, N]
    T* __restrict__ out_vals,         // shape: [batch_size, K]
    int* __restrict__ out_ids,        // shape: [batch_size, K]
    int N
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const T* row_input = input + row * N;
    T* row_out_vals = out_vals + row * K;
    int* row_out_ids = out_ids + row * K;

    T local_vals[K];
    int local_ids[K];

    #pragma unroll
    for (int i = 0; i < K; i++) {
        local_vals[i] = -1e10f;
        local_ids[i]  = -1;
    }

    // Each thread processes elements with stride = blockDim.x
    for (int i = tid; i < N; i += blockDim.x) {
        T val = row_input[i];
        insert_to_local_rank<T, K>(local_vals, local_ids, val, i);
    }

    extern __shared__ char shared_mem[];
    T* smem_vals = reinterpret_cast<T*>(shared_mem);
    int* smem_ids = reinterpret_cast<int*>(smem_vals + blockDim.x * K);

    // Write local results to shared memory
    #pragma unroll
    for (int i = 0; i < K; i++) {
        smem_vals[tid * K + i] = local_vals[i];
        smem_ids[tid * K + i]  = local_ids[i];
    }

    __syncthreads();

    // Thread 0 merges all local results
    if (tid == 0) {
        T final_vals[K];
        int final_ids[K];

        #pragma unroll
        for (int i = 0; i < K; i++) {
            final_vals[i] = -1e38f;
            final_ids[i]  = -1;
        }

        for (int i = 0; i < blockDim.x * K; i++) {
            insert_to_local_rank<T, K>(final_vals, final_ids, smem_vals[i], smem_ids[i]);
        }

        #pragma unroll
        for (int i = 0; i < K; i++) {
            row_out_vals[i] = final_vals[i];
            row_out_ids[i] = final_ids[i];
        }
    }
}

// ============================================================================
// Reference CPU TopK for validation
// ============================================================================

template<typename T>
void topk_cpu_reference(const T* input, T* out_vals, int* out_ids, int N, int K) {
    std::vector<std::pair<T, int>> indexed(N);
    for (int i = 0; i < N; i++) {
        indexed[i] = {input[i], i};
    }

    std::partial_sort(indexed.begin(), indexed.begin() + K, indexed.end(),
        [](const std::pair<T, int>& a, const std::pair<T, int>& b) {
            return a.first > b.first;
        });

    for (int i = 0; i < K; i++) {
        out_vals[i] = indexed[i].first;
        out_ids[i] = indexed[i].second;
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void print_device_info() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "========================================" << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max Threads per MP: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Memory Clock Rate: " << prop.memoryClockRate * 1e-6 << " GHz" << std::endl;
    std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bit" << std::endl;

    // Theoretical bandwidth
    float peak_bw = (prop.memoryClockRate * 1e3) * (prop.memoryBusWidth / 8.0) / 1e9;
    std::cout << "  Theoretical Peak Bandwidth: " << peak_bw << " GB/s" << std::endl;
    std::cout << "========================================" << std::endl;
}

// ============================================================================
// Benchmark Function - K is compile-time constant
// ============================================================================

template<typename T, int K>
void benchmark_topk(int batch_size, int N, int num_warmup, int num_iters) {
    const int threads = 256;
    const size_t input_size = batch_size * N;
    const size_t output_size = batch_size * K;

    // Allocate host memory
    std::vector<T> h_input(input_size);
    std::vector<T> h_out_vals(output_size);
    std::vector<int> h_out_ids(output_size);
    std::vector<T> h_out_vals_ref(output_size);
    std::vector<int> h_out_ids_ref(output_size);

    // Initialize with random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(-1000.0f, 1000.0f);
    for (size_t i = 0; i < input_size; i++) {
        h_input[i] = dist(gen);
    }

    // Allocate device memory
    T *d_input, *d_out_vals;
    int *d_out_ids;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out_vals, output_size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out_ids, output_size * sizeof(int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(T), cudaMemcpyHostToDevice));

    // Calculate shared memory size
    size_t smem_size = (threads * K * sizeof(T)) + (threads * K * sizeof(int));

    // Warmup
    for (int i = 0; i < num_warmup; i++) {
        topk_kernel_small_k<T, K><<<batch_size, threads, smem_size>>>(
            d_input, d_out_vals, d_out_ids, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        topk_kernel_small_k<T, K><<<batch_size, threads, smem_size>>>(
            d_input, d_out_vals, d_out_ids, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_out_vals.data(), d_out_vals, output_size * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_ids.data(), d_out_ids, output_size * sizeof(int), cudaMemcpyDeviceToHost));

    // Validate results (only for small problem sizes)
    bool validated = true;
    if (N <= 10000 && batch_size <= 100) {
        for (int b = 0; b < batch_size; b++) {
            topk_cpu_reference(h_input.data() + b * N,
                              h_out_vals_ref.data() + b * K,
                              h_out_ids_ref.data() + b * K, N, K);
        }

        // Compare
        for (size_t i = 0; i < output_size; i++) {
            if (h_out_vals[i] != h_out_vals_ref[i]) {
                validated = false;
                std::cout << "Validation mismatch at index " << i << std::endl;
                break;
            }
        }
    }

    // Calculate bandwidth
    // Read: input (batch_size * N * sizeof(T))
    // Write: out_vals + out_ids (batch_size * K * (sizeof(T) + sizeof(int)))
    float total_read_bytes = batch_size * N * sizeof(T);
    float total_write_bytes = batch_size * K * (sizeof(T) + sizeof(int));
    float total_bytes = total_read_bytes + total_write_bytes;
    float avg_time_ms = elapsed_ms / num_iters;
    float bandwidth_gbs = (total_bytes / avg_time_ms) / 1e6;
    float throughput_gbps = (total_bytes * 8.0 / avg_time_ms) / 1e6;

    // Print results
    std::cout << "TopK Benchmark (K=" << K << ", dtype=float32)" << std::endl;
    std::cout << "  Input shape: [" << batch_size << ", " << N << "]" << std::endl;
    std::cout << "  Output shape: [" << batch_size << ", " << K << "]" << std::endl;
    std::cout << "  Threads per block: " << threads << std::endl;
    std::cout << "  Shared memory: " << smem_size << " bytes" << std::endl;
    std::cout << "  Time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  Bandwidth: " << bandwidth_gbs << " GB/s" << std::endl;
    std::cout << "  Throughput: " << throughput_gbps << " Gbps" << std::endl;
    std::cout << "  Validation: " << (validated ? "PASSED" : "SKIPPED/FAILED") << std::endl;
    std::cout << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_out_vals));
    CUDA_CHECK(cudaFree(d_out_ids));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================================================
// Dispatcher - dispatch to compile-time K
// ============================================================================

template<typename T>
void dispatch_topk(int K, int batch_size, int N, int num_warmup, int num_iters) {
    switch(K) {
        case 4:
            benchmark_topk<T, 4>(batch_size, N, num_warmup, num_iters);
            break;
        case 8:
            benchmark_topk<T, 8>(batch_size, N, num_warmup, num_iters);
            break;
        case 16:
            benchmark_topk<T, 16>(batch_size, N, num_warmup, num_iters);
            break;
        case 32:
            benchmark_topk<T, 32>(batch_size, N, num_warmup, num_iters);
            break;
        default:
            std::cout << "Unsupported K=" << K << std::endl;
            break;
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    print_device_info();

    int num_warmup = 10;
    int num_iters = 100;

    std::cout << std::endl;

    // Test different configurations
    std::vector<int> K_values = {4, 8, 16, 32};
    std::vector<std::pair<int, int>> configs = {
        {1, 1024},
        {1, 4096},
        {1, 16384},
        {1, 65536},
        {1, 262144},
        {32, 1024},
        {32, 4096},
        {32, 16384},
        {64, 65536},
        {128, 131072},
    };

    for (int K : K_values) {
        std::cout << "========================================" << std::endl;
        std::cout << "Testing K = " << K << std::endl;
        std::cout << "========================================" << std::endl;

        for (auto [batch_size, N] : configs) {
            // Skip if shared memory would be too large
            size_t smem_needed = 256 * K * sizeof(float) + 256 * K * sizeof(int);
            if (smem_needed > 49152) { // 48KB typical limit
                std::cout << "Skipping [" << batch_size << ", " << N
                          << "] - shared memory (" << smem_needed
                          << " bytes) exceeds typical limit" << std::endl;
                continue;
            }

            dispatch_topk<float>(K, batch_size, N, num_warmup, num_iters);
        }
        std::cout << std::endl;
    }

    return 0;
}
