# cutlass入门

## 1. 存储方式

在以往的开发中，我们常常需要手动维护两片的空间，一部分在GPU上，一部分在CPU上，两部分通过内存拷贝的形式进行交互，但是其实际是维护着同样一份的数据

在cutlass中我们可以使用一个TensorRef来同时维护这两份数据，我们使用HostTensor即可创建一个维护两份数据的工具类，HostTensor中有两个模板参数:数据类型和主序方向

使用`device_ref`可以获取到在设备端的对象，使用`device_data`获取设备端的元素存储位置

```cuda
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A({M, K});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({K, N});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C({M, N});
```

`sync_device` 则是将数据在设备和主机之间进行一个同步

如下所示调用`gemm_op`的接口进行计算，返回`status`

```cuda
    status = gemm_op({
        {M, N, K},
        {ptrA, lda},
        {ptrB, ldb},
        {ptrC, ldc},
        {ptrD, ldd},
        {alpha, beta}
    });
```