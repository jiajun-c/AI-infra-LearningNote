# cutlass epilogue

使用epilogue fusion的作用是避免对全局地址的冗余访存，在对应部分结果出来的同时做一些elementwise的操作比如说relu

## 1. 基础的使用

对于 C = A * B + C这种场景而言，其在计算完A * B之后同样需要使用一个epilogue的阶段去累加结果到C上面

```cpp

```

