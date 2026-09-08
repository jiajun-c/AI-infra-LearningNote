# GPU算法

## 1. 排序算法

### 1.1 双调排序

每一步构建一段上升+下降，重复这个过程，让大小从2一直二次幂增长到len，直到所有的都完成构建。

```cpp
__global__ void sort(int* values, int len) {
    for (int k = 2; k <= len; k <<= 1) {
        for (int j = kk >> 1; j > 0; j >>= 1) {
            for (int idx = threadIdx.x; idx < len; idx += blockDim.x) {
                int p = idx ^ j;
                if (p > idx) {
                    bool keepmin = ((idx & k) == 0);
                    int a = values[idx], b = values[p];
                    if ((a > b) == keepmin) {
                        values[idx] = b;
                        values[p]   = a;
                    }
                }
            }
            __syncthreads();
        }
    } 
}
```

### 1.2 基数排序

对32个bit位按照递增的顺序进行排序，让为0的在前面，让为1的在后面。

假设需要对float进行排序，那么将符号位给取反即可。

```cpp
__global__ void radixsort(unsigned int* v) {
    __shared__ unsigned int key[N];
    __shared__ unsigned int scan[N];
    const int t = threadIdx.x;
    key[t] = v[t];
    __syncthreads();
    for (int bit = 0; bit < 32; bit++) {
        const unsigned int now = key[t];
        const unsigned int b = (x >> bit) & 1u;
        scan[t] = 1 - b;
        __syncthreads();
        for (int d = 1; d < N; d <<= 1) {
            int add = (t >= d) ? scan[t - d]: 0u;
            __syncthreads();
            scan[t] += add;
            __syncthreads();
        }
        int totalZeros = scan[N-1];
        int zeroBefore = scan[t] - (1u - b);
        int onesBefore = t - zeroBefore;
        int dst = (b == 0u) ? zeroBefore: totalZeros + onesBefore;
        __syncthreads();
        key[dst] = now;
        __syncthreads();
    }
    v[t] = key[t]; 
}
```

## 2. topk 算法

## 3. 前缀和

