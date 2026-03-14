# element wise 优化

## baseline

```shell
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         6.99
    SM Frequency                    Ghz         1.54
    Elapsed Cycles                cycle    1,433,114
    Memory Throughput                 %        92.36
    DRAM Throughput                   %        92.36
    Duration                         us       927.65
    L1/TEX Cache Throughput           %        24.43
    L2 Cache Throughput               %        18.30
    SM Active Cycles              cycle 1,430,556.54
    Compute (SM) Throughput           %        24.39
    ----------------------- ----------- ------------
```

## pack 优化

```shell
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         6.99
    SM Frequency                    Ghz         1.54
    Elapsed Cycles                cycle    1,365,199
    Memory Throughput                 %        93.13
    DRAM Throughput                   %        93.13
    Duration                         us       883.71
    L1/TEX Cache Throughput           %        16.65
    L2 Cache Throughput               %        27.81
    SM Active Cycles              cycle 1,360,623.29
    Compute (SM) Throughput           %         2.63
    ----------------------- ----------- ------------
```