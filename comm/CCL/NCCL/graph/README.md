# NCCL cuda图接口

之前在cuda部分我们发现了图接口的好处，可以将若干kernel进行图捕获从而减少CPU侧的kernel launch开销。

NCCL中的qi始支持了cuda图接口，但是需要保证捕获时的一致性，当rank0在录制时，其他rank的节点也需要处于录制状态。

