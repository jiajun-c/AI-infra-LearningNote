# NCCL修改配置

## 1. 配置CTA数量

在nccl如reduce等操作其实都是借助sm进行实现的，而计算的kernel也需要sm，而GPU上sm的数量是有限的，占用的sm数量越多那么计算kernel的性能影响也会更大

在不配置CTA数量的情况下，nccl自动选择，初始化时直接使用`ncclCommInitAll` 即可

```cpp
ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * nGPUs);
    ncclCommInitAll(comms, nGPUs, devlist);
```

如果需要修改CTA的数量， 需要传入config参数进行初始化，如下所示，设置使用4个CTA     你

```cpp
ncclUniqueId id;
NCCLCHECK(ncclGetUniqueId(&id));

ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
config.blocking = 1;     // 保持阻塞模式，与 InitAll 行为一致
config.maxCTAs = 4;      // <--- 核心修改：强制最大 4 个 SM
config.minCTAs = 4;      // (可选) 强制最小也为 4，锁定占用量

printf("Initializing NCCL with maxCTAs = %d ...\n", config.maxCTAs);

NCCLCHECK(ncclGroupStart());
for (int i = 0; i < nGPUs; i++) {
    CUDACHECK(cudaSetDevice(devlist[i]));
    NCCLCHECK(ncclCommInitRankConfig(comms + i, nGPUs, id, i, &config));
}
NCCLCHECK(ncclGroupEnd());
```

测试allReduce在8卡H200上的性能如下所示，当我们限制所使用的sm的数量后，性能下降明显

```shell
=== Performance Results (CTA Limit: 8) ===
Data Size:      128.00 MB
Time:           1.003 ms
Alg Bandwidth:  133.80 GB/s
Bus Bandwidth:  234.15 GB/s
==========================================

=== Performance Results (CTA Limit: 4) ===
Data Size:      128.00 MB
Time:           1.759 ms
Alg Bandwidth:  76.31 GB/s
Bus Bandwidth:  133.54 GB/s
==========================================

=== Performance Results (CTA Limit: 2) ===
Data Size:      128.00 MB
Time:           3.169 ms
Alg Bandwidth:  42.35 GB/s
Bus Bandwidth:  74.11 GB/s
==========================================

=== Performance Results (CTA Limit: 1) ===
Data Size:      128.00 MB
Time:           6.262 ms
Alg Bandwidth:  21.43 GB/s
Bus Bandwidth:  37.51 GB/s
==========================================

Verifying results...
Verification Passed. 解释一下这个性能
```

## 2. 