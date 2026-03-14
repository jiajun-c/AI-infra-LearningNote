# NCCL

NCCL 是英伟达的通信库，用于进行集合通信。

## 1. 初始化通信域

单核单线程使用多个 GPU

如下所示进行设备的初始化

```c
  ncclComm_t comms[4];
  int nDev = 4;
  int devs[4] = { 0, 1, 2, 3 };
  ncclCommInitAll(comms, nDev, devs);
```

初始化通信域，使用更加底层的代码编写如下所示，使用 `ncclCommInitRank`

```c
ncclResult_t ncclCommInitAllMy(ncclComm_t* comm, int ndev, const int* devlist) {
    ncclUniqueId Id;
    ncclGetUniqueId(&Id);
    ncclGroupStart();
    for (int i=0; i<ndev; i++) {
      cudaSetDevice(devlist[i]);
      ncclCommInitRank(comm+i, ndev, Id, i);
    }
    ncclGroupEnd();
}
```

也可以对通信域进行切分，这点和 MPI 很像，如下所示

```c
int rank, nranks;
ncclCommUserRank(comm, &rank);
ncclCommCount(comm, &nranks);
ncclCommSplit(comm, rank/(nranks/2), rank%(nranks/2), &newcomm, NULL);
```

## 2. 通信原语

### 2.1 AllReduce

使用 AllReduce 可以进行 reduceSum，reduceMax 等操作，最终每个节点中都有一份最终输出的数据

```c
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i)
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
        comms[i], s[i]));
  NCCLCHECK(ncclGroupEnd());
```

### 2.2 Bcast

Bcast 可以将数据从一个节点广播到所有的节点，如下所示

```c
      for (int i = 0; i < nDev; ++i)
        NCCLCHECK(ncclBroadcast((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, 0,
            comms[i], s[i]));
      NCCLCHECK(ncclGroupEnd());
```

### 2.3 AllGather

AllGather 可以将多个节点的数据合并并且放到全部的节点上

```c
// 每个节点发送自己的数据，接收所有节点的数据
NCCLCHECK(ncclGroupStart());
for (int i = 0; i < nDev; ++i)
  NCCLCHECK(ncclAllGather((const void*)sendbuff[i], (void*)recvbuff[i],
                          sendcount, ncclFloat, comms[i], s[i]));
NCCLCHECK(ncclGroupEnd());
```

### 2.4 Reduce-Scatter

Reduce-Scatter 先进行 reduce 操作，然后将结果分散到各个节点：

```c
NCCLCHECK(ncclReduceScatter((const void*)sendbuff, (void*)recvbuff,
                            recvcount, ncclFloat, ncclSum, comm, stream));
```

### 2.5 AllToAll

AllToAll 允许每个节点向其他所有节点发送不同的数据：

```c
NCCLCHECK(ncclGroupStart());
for (int i = 0; i < nDev; ++i)
  NCCLCHECK(ncclAllToAll((const void*)sendbuff[i], (void*)recvbuff[i],
                         count, ncclFloat, comms[i], s[i]));
NCCLCHECK(ncclGroupEnd());
```

## 3. 点对点通信 (P2P)

点对点通信和 MPI 很相似，使用 `ncclSend` 和 `ncclRecv` 进行通信。

```c
// 发送数据到指定 peer
NCCLCHECK(ncclSend((const void*)sendbuff, count, ncclFloat,
                   peer, comm, stream));

// 从指定 peer 接收数据
NCCLCHECK(ncclRecv((void*)recvbuff, count, ncclFloat,
                   peer, comm, stream));
```

### 3.1 环形通信模式

在流水线并行中常用环形通信：

```
Rank 0 -> Rank 1 -> Rank 2 -> Rank 3 -> Rank 0
```

```c
int prev_rank = (rank - 1 + nranks) % nranks;
int next_rank = (rank + 1) % nranks;

// 从上一个节点接收
ncclRecv(recvbuf, count, ncclFloat, prev_rank, comm, stream);
// 向下一个节点发送
ncclSend(sendbuf, count, ncclFloat, next_rank, comm, stream);
```

## 4. NCCL 拓扑感知

NCCL 会自动检测 GPU 拓扑结构（NVLink、PCIe 等）并选择最优的通信路径。

### 4.1 拓扑检测

```c
ncclTopoPrint(ncclComm_t comm);
```

### 4.2 手动设置拓扑

通过环境变量控制：
- `NCCL_TOPO_FILE`: 指定拓扑文件
- `NCCL_NET_GDR_LEVEL`: 设置 GPUDirect RDMA 级别

## 5. 性能优化

### 5.1 带宽优化

- 使用 NVLink/NVSwitch 连接
- 启用 GPUDirect RDMA
- 选择合适的数据类型

### 5.2 延迟优化

- 批量通信操作
- 使用异步通信
- 通信计算重叠
