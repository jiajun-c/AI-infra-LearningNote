# NCCL buffer 优化

NCCL可以提前注册部分空间为buffer，从而建立TMA的映射。在进行通信的时候可以减少sm的开销，直接交给CE去做，更利于计算和通信的重叠。

在没有buffer的情况下，则是需要SM将数据拷贝到buffer中，然后再进行通信，这是之前通信和计算无法很好的重叠的原因之一

如果说我们仅考虑通信时间的话其实两个时间是差不多，需要结合计算时间来一起看

其最佳的场景是用于一个memory bound的算子+通信的情况下，使用buffer可以有效减少对SM的占用，使得更多的时间可以用于memory bound算子的计算上

```cpp
CHECK(ncclMemAlloc(&sendbuff, count * sizeof(float)));
CHECK(ncclMemAlloc(&recvbuff, count * sizeof(float)));

CHECK(ncclCommRegister(comm, sendbuff, count * sizeof(float), &sendRegHandle));
CHECK(ncclCommRegister(comm, recvbuff, count * sizeof(float), &recvRegHandle));

CHECK(ncclAllReduce(sendbuff, recvbuff, 1024, ncclFloat, ncclSum, comm, stream));
CHECK(ncclAllReduce((void*)((float*)sendbuff + 1024), (void*)((float*)recvbuff + 2048), 1024, ncclFloat, ncclSum, comm, stream));
CHECK(cudaStreamSynchronize(stream));

CHECK(ncclCommDeregister(comm, sendRegHandle));
CHECK(ncclCommDeregister(comm, recvRegHandle));

CHECK(ncclMemFree(sendbuff));
CHECK(ncclMemFree(recvbuff));
```


