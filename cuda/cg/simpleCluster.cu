#include <cstdio>
#define _CG_HAS_CLUSTER_GROUP

#include <cooperative_groups.h>
#include <iostream>

using namespace std;

namespace cg = cooperative_groups;

__global__ void __cluster_dims__(2, 1, 1) 
simple_kernel() {
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int trank = cluster.thread_rank();
    int thread_count = cluster.num_threads();
    int block_count = cluster.num_blocks();
    unsigned int rank = cluster.block_rank();
    if (threadIdx.x == 0) {
        printf("thread count %d block count %d\n", thread_count, block_count);
        printf("block rank %d %d\n", rank, trank);
    }
}

int main() {

    simple_kernel<<<4, 32>>>();
    cudaDeviceSynchronize();
}