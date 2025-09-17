#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/random.h>
using namespace std;

template <typename T>
struct minmax_pair
{
  T min_val;
  T max_val;
};

template <typename T>
struct minmax_unary_op: public thrust::unary_function<T, minmax_pair<T> >
{
    __host__ __device__ 
    minmax_pair<T> operator()(const T& x) const
    {
        minmax_pair<T> result;
        result.min_val = x;
        result.max_val = x;
        return result;
    }
};

template <typename T>
struct minmax_binary_op: public thrust::binary_function<minmax_pair<T>, T, minmax_pair<T> >
{
    __host__ __device__
    minmax_pair<T> operator()(const minmax_pair<T>& x, const minmax_pair<T>& y) const
    {
      minmax_pair<T> result;
      result.min_val = thrust::min(x.min_val, y.min_val);
      result.max_val = thrust::max(x.max_val, y.max_val);
      return result;
    }
};


int main() {
    size_t N = 10;

    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(10, 99);

    thrust::host_vector<int> data(N);
    for (size_t i = 0; i < N; i++)
      data[i] = dist(rng);
    minmax_unary_op<int>  unary_op;
    minmax_binary_op<int> binary_op;
    minmax_pair<int> init = unary_op(data[0]);
    minmax_pair<int> result = thrust::transform_reduce(data.begin(), data.end(), unary_op, init, binary_op);
    std::cout << "[ ";
    for(size_t i = 0; i < N; i++)
    std::cout << data[i] << " ";
    std::cout << "]" << std::endl;
   
    std::cout << "minimum = " << result.min_val << std::endl;
    std::cout << "maximum = " << result.max_val << std::endl;
  
    return 0;
  }