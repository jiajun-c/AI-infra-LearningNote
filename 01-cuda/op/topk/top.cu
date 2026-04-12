#include <cuda.h>
#include <cuda_runtime.h>

// 5 4 3 2 1
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

template<typename T, int K>
__global__ void topk_kernel_small_k(
    const T* __restrict__ input,  // shape: [batch_size, N]
    T* __restrict__ out_vals,     // shape: [batch_size, K]
    int* __restrict__ out_ids,    // shape: [batch_size, K]
    int N
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const T* row_input = input + row * N;
    T* row_out_vals = out_vals + row * K;
    int* row_out_ids =  out_ids + row * K;

    T local_vals[K];
    int local_ids[K];

    #pragma unroll
    for (int i = 0; i < K; i++) {
        local_vals[i] = -1e10f;
        local_ids[i]  = -1;
    }

    for (int i = tid; i < N; i += blockDim.x) {
        T val = row_input[i];
        insert_to_local_rank<T, K>(local_vals, local_ids, val, i);
    }

    extern __shared__ char shared_mem[];
    T* smem_vals = reinterpret_cast<T*>(shared_mem);
    int* smem_ids = reinterpret_cast<int*>(smem_vals + blockDim.x * K * sizeof(T));

    #pragma unroll
    for (int i = 0; i < K; i++) {
        smem_vals[tid * K + i] = local_vals[i];
        smem_ids[tid * K + i]  = local_ids[i];
    }

    __syncthreads();

    if (tid == 0) {
        T final_vals[K];
        int final_ids[K];

        #pragma unroll
        for (int i = 0; i < K; i++) {
            final_vals[i] = -1e38f;
            final_ids[i] = -1;
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