#include <cstdio>
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

__device__ void print_arch(){
  const char my_compile_time_arch[] = STR(__CUDA_ARCH__);
  printf("__CUDA_ARCH__: %s\n", my_compile_time_arch);
}
__global__ void example()
{
   print_arch();
}

int main(){
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    printf("__CUDA_ARCH__ >= 700\n");
    // 使用 Tensor Core 进行矩阵乘法的代码
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ...");
#endif
example<<<1,1>>>();
cudaDeviceSynchronize();
}