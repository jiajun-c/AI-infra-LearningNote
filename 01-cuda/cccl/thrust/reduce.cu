#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <iostream>

using namespace std;
int my_rand(void)
{
  static thrust::default_random_engine rng;
  static thrust::uniform_int_distribution<int> dist(0, 9999);
  return dist(rng);
}
int main() {
    thrust::host_vector<int> h_vec(10);
    thrust::generate(h_vec.begin(), h_vec.end(), my_rand);
    thrust::device_vector<int> d_vec = h_vec;
    int init = 0;
    thrust::plus<int> binary_op; 
    int sum = thrust::reduce(d_vec.begin(), d_vec.end(), init, binary_op);
    std::cout << "sum: " << sum << std::endl;
    return 0;
}