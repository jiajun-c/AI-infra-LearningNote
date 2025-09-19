# CUDA 架构/编译


## 1. CUDA 架构演进

### 1.1 Tesla架构(2006)

首次支持C语言编程，使得GPU可以进行通用计算


### 1.2 Fermi架构(2010)

### 1.3 Kepler架构(2012)

### 1.4 Maxwell架构(2014)

### 1.5 Pascal架构(2015)

### 1.6 Volta架构(2017)

### 1.7 Turing架构(2018)

### 1.8 Ampere架构(2020)

### 1.9 Hopper架构(2022)

### 1.10 BlackWell 架构(2023)

在device侧中可以使用`__CUDA_ARCH__` 获取到当前的架构版本，例如`compute_80` 等于 __CUDA_ARCH__ 为 800。可以通过该信息来进行条件编译。

```cpp
__device__ void print_arch(){
  const char my_compile_time_arch[] = STR(__CUDA_ARCH__);
  printf("__CUDA_ARCH__: %s\n", my_compile_time_arch);
}
__global__ void example()
{
   print_arch();
}
```