# cuda gemv 算子优化

gemv使用一个矩阵乘以一个向量的形式，对于gemv的计算，有两种形式

- 一个线程计算一行矩阵
- 一个线程块计算一行矩阵
- 一个线程块计算多行矩阵

在A100上进行测试，我们可以发现 Warp1-Naive 的实现是最快的。对于向量的共享内存优化效果不明显。

```shell

[HGEMV 2025-09-29 14:41:34 613688:613688 tester.h:52 evaluate] ----------------- Evaluating Cublas-Tensor-Op -----------------
[HGEMV 2025-09-29 14:41:36 613688:613688 tester.h:62 evaluate] Warm up time: 1870.640 ms
[HGEMV 2025-09-29 14:41:36 613688:613688 tester.h:102 profile] Cublas-Tensor-Op exit, profiling time: 0.0154 ms (100.00%), throughput: 0.002133 TFLOPS (100.00%)
[HGEMV 2025-09-29 14:41:36 613688:613688 tester.h:52 evaluate] ----------------- Evaluating Thread-Naive -----------------
[HGEMV 2025-09-29 14:41:36 613688:613688 tester.h:62 evaluate] Warm up time: 0.302 ms
[HGEMV 2025-09-29 14:41:36 613688:613688 tester.h:102 profile] Thread-Naive exit, profiling time: 0.0314 ms (204.67%), throughput: 0.001042 TFLOPS (48.86%)
[HGEMV 2025-09-29 14:41:36 613688:613688 tester.h:52 evaluate] ----------------- Evaluating Thread-Smem -----------------
[HGEMV 2025-09-29 14:41:37 613688:613688 thread_smem.cu:46 initThreadSmem] smem_max_size: 0 KBytes (256 bytes)
[HGEMV 2025-09-29 14:41:37 613688:613688 tester.h:62 evaluate] Warm up time: 428.129 ms
[HGEMV 2025-09-29 14:41:37 613688:613688 tester.h:102 profile] Thread-Smem exit, profiling time: 0.0282 ms (183.33%), throughput: 0.001164 TFLOPS (54.55%)
[HGEMV 2025-09-29 14:41:37 613688:613688 tester.h:52 evaluate] ----------------- Evaluating Warp1-Naive -----------------
[HGEMV 2025-09-29 14:41:37 613688:613688 tester.h:62 evaluate] Warm up time: 0.292 ms
[HGEMV 2025-09-29 14:41:37 613688:613688 tester.h:102 profile] Warp1-Naive exit, profiling time: 0.0072 ms (46.67%), throughput: 0.004571 TFLOPS (214.29%)
[HGEMV 2025-09-29 14:41:37 613688:613688 tester.h:52 evaluate] ----------------- Evaluating Warp1-Smem -----------------
[HGEMV 2025-09-29 14:41:37 613688:613688 warp1_smem.cu:61 initWarp1Smem] smem_max_size: 0 KBytes (256 bytes)
[HGEMV 2025-09-29 14:41:37 613688:613688 tester.h:62 evaluate] Warm up time: 459.177 ms
[HGEMV 2025-09-29 14:41:37 613688:613688 tester.h:102 profile] Warp1-Smem exit, profiling time: 0.0073 ms (47.33%), throughput: 0.004507 TFLOPS (211.27%)
[HGEMV 2025-09-29 14:41:37 613688:613688 tester.h:52 evaluate] ----------------- Evaluating Warp2-Naive -----------------
[HGEMV 2025-09-29 14:41:37 613688:613688 tester.h:62 evaluate] Warm up time: 0.275 ms
[HGEMV 2025-09-29 14:41:37 613688:613688 tester.h:102 profile] Warp2-Naive exit, profiling time: 0.0074 ms (48.00%), throughput: 0.004444 TFLOPS (208.33%)
[HGEMV 2025-09-29 14:41:37 613688:613688 tester.h:52 evaluate] ----------------- Evaluating Warp2-Smem -----------------
[HGEMV 2025-09-29 14:41:38 613688:613688 warp2_smem.cu:65 initWarp2Smem] smem_max_size: 0 KBytes (256 bytes)
[HGEMV 2025-09-29 14:41:38 613688:613688 tester.h:62 evaluate] Warm up time: 223.054 ms
[HGEMV 2025-09-29 14:41:38 613688:613688 tester.h:102 profile] Warp2-Smem exit, profiling time: 0.0075 ms (48.67%), throughput: 0.004384 TFLOPS (205.48%)
[HGEMV 2025-09-29 14:41:38 613688:613688 tester.h:52 evaluate] ----------------- Evaluating Warp4-Naive -----------------
[HGEMV 2025-09-29 14:41:38 613688:613688 tester.h:62 evaluate] Warm up time: 0.274 ms
[HGEMV 2025-09-29 14:41:38 613688:613688 tester.h:102 profile] Warp4-Naive exit, profiling time: 0.0076 ms (49.33%), throughput: 0.004324 TFLOPS (202.70%)
[HGEMV 2025-09-29 14:41:38 613688:613688 tester.h:52 evaluate] ----------------- Evaluating Warp4-Smem -----------------
[HGEMV 2025-09-29 14:41:38 613688:613688 warp4_smem.cu:65 initWarp4Smem] smem_max_size: 0 KBytes (256 bytes)
[HGEMV 2025-09-29 14:41:38 613688:613688 tester.h:62 evaluate] Warm up time: 26.974 ms
[HGEMV 2025-09-29 14:41:38 613688:613688 tester.h:102 profile] Warp4-Smem exit, profiling time: 0.0077 ms (50.00%), throughput: 0.004267 TFLOPS (200.00%)
[HGEMV 2025-09-29 14:41:38 613688:613688 tester.h:52 evaluate] ----------------- Evaluating Warp8-Naive -----------------
[HGEMV 2025-09-29 14:41:38 613688:613688 tester.h:62 evaluate] Warm up time: 0.279 ms
[HGEMV 2025-09-29 14:41:38 613688:613688 tester.h:102 profile] Warp8-Naive exit, profiling time: 0.0091 ms (59.33%), throughput: 0.003596 TFLOPS (168.54%)
[HGEMV 2025-09-29 14:41:38 613688:613688 tester.h:52 evaluate] ----------------- Evaluating Warp8-Smem -----------------
[HGEMV 2025-09-29 14:41:39 613688:613688 warp8_smem.cu:65 initWarp8Smem] smem_max_size: 0 KBytes (256 bytes)
[HGEMV 2025-09-29 14:41:39 613688:613688 tester.h:62 evaluate] Warm up time: 446.463 ms
[HGEMV 2025-09-29 14:41:39 613688:613688 tester.h:102 profile] Warp8-Smem exit, profiling time: 0.0090 ms (58.67%), throughput: 0.003636 TFLOPS (170.45%)
[HGEMV 2025-09-29 14:41:39 613688:613688 tester.h:52 evaluate] ----------------- Evaluating Warp16-Naive -----------------
[HGEMV 2025-09-29 14:41:39 613688:613688 tester.h:62 evaluate] Warm up time: 0.264 ms
[HGEMV 2025-09-29 14:41:39 613688:613688 tester.h:102 profile] Warp16-Naive exit, profiling time: 0.0124 ms (80.67%), throughput: 0.002645 TFLOPS (123.97%)
[HGEMV 2025-09-29 14:41:39 613688:613688 tester.h:52 evaluate] ----------------- Evaluating Warp16-Smem -----------------
[HGEMV 2025-09-29 14:41:39 613688:613688 warp16_smem.cu:65 initWarp16Smem] smem_max_size: 0 KBytes (256 bytes)
[HGEMV 2025-09-29 14:41:39 613688:613688 tester.h:62 evaluate] Warm up time: 213.685 ms
[HGEMV 2025-09-29 14:41:39 613688:613688 tester.h:102 profile] Warp16-Smem exit, profiling time: 0.0124 ms (80.67%), throughput: 0.002645 TFLOPS (123.97%)
[HGEMV 2025-09-29 14:41:39 613688:613688 main.cu:106 main] Done
```