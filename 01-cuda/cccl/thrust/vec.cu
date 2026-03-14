#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

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

