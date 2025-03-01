#include <cooperative_groups.h>
#include <cuda.h>

__global__ void clusterHist_kernel(int *bins, const int nbins, 
                                    const int bins_per_block,
                                    cont int *__restrict__ input,
                                    size_t arrary_size) {
    extern __shared__ int smem[];
    namespace cg = cooperative_groups;
    int tid = cg::this_grid().thread_rank()

    cg::cluster_group cluster = cg::this_cluster();
    int cluster_size = cluster.dim_blocks().x;
    for (int i = threadIdx.x; i < bin_per_block, i += blockDim.x) {
        smem[i] = 0;
    }
    cluster.sync();
    for (int i = tid; i < arrary_size; i += blockDim.x * gridDim.x) {
        int ldata = input[i];
        int binid = ldata;
        if (ldata < 0) binid = 0;
        else if (ldata >= nbins) binid = nbins - 1;

        int dst_block_rank = (int)(binid / bins_per_block);
        int dst_offset = binid % bins_per_block;
        int *dst_smeme = cluster.map_shared_rank(smem, dst_block_rank);
        atomicAdd(dst_smem + dst_offset, 1);
    }
    cluster.sync(); 
    int *lbins = bins + cluster.block_rank() * bins_per_block;
    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
        atomicAdd(&lbins[i], smem[i]);
    }
}

int main() {
    cudaLaunchConfig_t config = {0};
    config.gridDim = 
}