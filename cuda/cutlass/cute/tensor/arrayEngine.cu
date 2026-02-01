#include <cstdio>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

using namespace cute;

// ... Layout 定义保持不变 ...
using LayoutA_Padded = Layout<Shape<Int<4>, Int<4>>, Stride<Int<1>, Int<5>>>;
using LayoutB_Dense = Layout<Shape<Int<4>, Int<4>>>;

struct SharedStorage {
    ArrayEngine<float, cosize_v<LayoutA_Padded>> A;
    ArrayEngine<float, cosize_v<LayoutB_Dense>>  B;
};

__global__ void verify_layout_kernel() {
    using namespace cute;
    extern __shared__ char smem_raw[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

    if (threadIdx.x == 0) {
        print(cosize_v<LayoutA_Padded>);
        printf("\n=== Static Layout Info ===\n");
        printf("Layout A (Padded):\n");
        print(LayoutA_Padded{}); printf("\n");
        
        // 修复点 1: 强制转换为 int
        printf("  Logical Size (size):   %d elements\n", (int)size(LayoutA_Padded{}));
        printf("  Physical Size (cosize): %d elements (includes padding)\n", (int)cosize(LayoutA_Padded{}));
        
        printf("\nLayout B (Dense):\n");
        print(LayoutB_Dense{}); printf("\n");
        
        // 修复点 2: 强制转换为 int
        printf("  Logical Size (size):   %d elements\n", (int)size(LayoutB_Dense{}));
        printf("  Physical Size (cosize): %d elements\n", (int)cosize(LayoutB_Dense{}));

        printf("\n=== Memory Address Info ===\n");
        float* ptrA = (float*)&smem.A;
        float* ptrB = (float*)&smem.B;
        
        long actual_offset = (char*)ptrB - (char*)ptrA;
        long expected_min_offset = cosize(LayoutA_Padded{}) * sizeof(float);
        
        printf("Base Address (smem): %p\n", smem_raw);
        printf("Address of A:        %p\n", ptrA);
        printf("Address of B:        %p (Actual Offset: %ld bytes)\n", ptrB, actual_offset);
        printf("Min Expected Offset: %ld bytes (from cosize)\n", expected_min_offset);
        
        // 修复验证逻辑：允许编译器进行内存对齐带来的 Padding
        if (actual_offset >= expected_min_offset) {
            printf(">> SUCCESS: B starts after A ends. (Padding of %ld bytes detected)\n", actual_offset - expected_min_offset);
        } else {
            printf(">> ERROR: Memory layout overlap! B starts inside A.\n");
        }
    }
}

int main() {
    int smem_size_bytes = sizeof(SharedStorage);    
    printf("Host: Launching kernel with %d bytes of Shared Memory...\n", smem_size_bytes);
    verify_layout_kernel<<<1, 32, smem_size_bytes>>>();
    cudaDeviceSynchronize();
    return 0;
}