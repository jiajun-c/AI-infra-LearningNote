# NCCL

NCCL是英伟达的通信库，用于进行集合通信。

## 1. 初始化通信域

单核单线程使用多个GPU

如下所示进行设备的初始化

```c
  ncclComm_t comms[4];
  int nDev = 4;
  int devs[4] = { 0, 1, 2, 3 };
  ncclCommInitAll(comms, nDev, devs);
```

初始化通信域，使用更加底层的代码编写如下所示，使用`ncclCommInitRank` 

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

也可以对通信域进行切分，这点和MPI很像，如下所示

```c
int rank, nranks;
ncclCommUserRank(comm, &rank);
ncclCommCount(comm, &nranks);
ncclCommSplit(comm, rank/(nranks/2), rank%(nranks/2), &newcomm, NULL);
```

## 2. 通信原语

### 2.1 AllReduce

使用AllReduce可以进行reduceSum，reduceMax等操作，最终每个节点中都有一份最终输出的数据

```c
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i)
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
        comms[i], s[i]));
  NCCLCHECK(ncclGroupEnd());
```

### 2.2 Bcast

Bcast 可以将数据从一个节点广播到所有的节点, 如下所示

```c
      for (int i = 0; i < nDev; ++i)
        NCCLCHECK(ncclBroadcast((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, 0,
            comms[i], s[i]));
      NCCLCHECK(ncclGroupEnd());
```

### 2.3 AllGather

AllGather 可以将多个节点的数据合并并且放到全部的节点上


## 3. 点对点通信

点对点通信这块其实和MPI也很相似，

