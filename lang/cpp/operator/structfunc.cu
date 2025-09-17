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