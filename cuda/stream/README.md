# Cuda Stream

cuda中有一个默认的stream，该stream会导致隐式的同步阻塞，使得我们希望并发的流变为同步， 为了解决该问题有两种方式

- 编译时增加`--default-stream per-thread`参数
- 创建流的时候设置为非阻塞

```shell
nvcc --default-stream per-thread my_program.cu -o my_program
```

```cuda
cudaStream_t stream;
// 关键在于这个标志：cudaStreamNonBlocking
// 它告诉运行时：这个流不要和 Default Stream 发生任何纠缠。
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

// 之后在这个 stream 中提交任务，它就不会被默认流卡住了
kernel<<<grid, block, 0, stream>>>(...);
```
