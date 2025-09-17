# Thrust 库

Thrust库是一个nvidia开源的一个高阶高性能并行库

## 1. Thrust 向量

thrust向量中分为两种，`host_vector` 和 `device_vector`，一个在host侧内存中，一个分配在device侧内存中，使用 `resize` 可以调整向量的大小，使用赋值语句可以直接进行数据在host和device之间的拷贝。

```cpp
int main()
{
    thrust::host_vector<int> h_vec(10); 
    h_vec.resize(20);
    for (int i = 0; i < h_vec.size(); i++) {
        h_vec[i] = i;
    }

    thrust::device_vector<int> d_vec = h_vec;
    for (int i = 0; i < d_vec.size(); i++) {
        std::cout << d_vec[i] << " ";
    }
}
```



## 2. 基本操作

### 2.1 thrust::transform

thrust::transform 是一个用于对向量进行操作的函数， 其可以传入一个一元对象或者二元对象，再可以传入一个函数对象，对向量中的元素进行操作。

如下所示对向量中的元素进行axpy的操作

```cpp
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;
struct saxpy_functor {
    const float a;
    saxpy_functor(float _a) : a(_a) {}
    __host__ __device__ // 同时可在主机和设备端执行
    float operator()(const float& x, const float& y) const {
        return a * x + y;
    }
};
int main()
{
    thrust::device_vector<float>x(4);
    thrust::device_vector<float>y(4);
    for (int i = 0; i < x.size(); i++) {
        x[i] = i;
        y[i] = i;
    }
    thrust::device_vector<float>res(x.size());
    float alpha = 2.0;

    thrust::transform(x.begin(), x.end(), y.begin(), res.begin(),saxpy_functor(alpha));
    thrust::host_vector<float>res_host = res;
    for(int i = 0; i < res_host.size(); i++) {
        cout << res_host[i] << " ";
    }
    return 0;
}
```