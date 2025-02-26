## 1. RedceSum

### 1.1 在warp层面进行ReduceSum

如下所示，使用 `__shfl_down_sync` 将后半部分的数据加入到前半部分的数据上，以此来递归地实现warp层面的ReduceSum

```cpp
template <typename T>
__inline__  __device__ int WarpReduceSum(T val) {
    for (int offset = (warpSize >> 1); offset > 0; offset >>=1) {
        val += __shfl_down_sync(0xffffffff, val, offset, warpSize);
    }
    return val;
}
```

### 1.2 在Block层面的ReduceSum

在每个Block内，先进行Warp层面的ReduceSum将每个warp内的数据放入到一个sharedMem中，然后再通过一次WarpLevel的ReduceSum将其合并为一个数

```cpp
template<typename T>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
    int laneid = threadIdx.x % warpSize;
    int warpid = threadIdx.x / warpSize;
    val = WarpReduceSum(val);
    __syncthreads();
    if (laneid == 0) {
        shared[warpid] = val;
    }
    __syncthreads();
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[laneid] : T(0);
    if (warpid == 0) {
      val = WarpReduceSum(val);
    }
    return val;
}
```