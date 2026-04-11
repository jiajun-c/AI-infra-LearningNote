# Torch stream机制

torch的stream机制是基于cuda的stream进行实现的，其区别主要是在内存分配上

原文的注释介绍如下所示，大致可以分为下面的几点
- 内存分配是block级别的，其会划分若干个block，block只可以在当前的stream进行重新分配
- 每次都会选择最小的block进行分配，当不满足分配的时候将会尝试使用cudaMalloc进行分配，不够的话再去进行释放之前cache的block，如果还是不行的话，那么释放掉全部的block重新分配
- 为了避免碎片过多，会限制进行碎片化的次数，不然可能会把一个大的block切的过碎
- 由于这个数据的分配是由stream进行的，所以如果想进行同步，需要使用`recordStream`


```shell
//
// Yet another caching allocator for CUDA device allocations.
//
// - Allocations are associated with a stream. Once freed, blocks can be
//   re-allocated on the same stream, but not on any other stream.
// - The allocator attempts to find the smallest cached block that will fit the
//   requested size. If the block is larger than the requested size, it may be
//   split. If no block is found, the allocator will delegate to cudaMalloc.
// - If the cudaMalloc fails, the allocator will attempt to free one cached
//   block of sufficient size that is not split and retry the allocation.
//   If this also fails, the allocator will attempt to free all cached blocks
//   that are not split and retry the allocation.
// - Large (>1MB) and small allocations are stored in separate pools.
//   Small requests are packed into 2MB buffers. Large requests will use the
//   smallest available free block or allocate a new block using cudaMalloc.
// - To reduce fragmentation, requests between 1MB and 10MB will allocate and
//   split a 20MB block, if no free block of sufficient size is available.
// - To further reduce fragmentation, blocks >= max_split_size are not allowed
//   to be split. These oversize cached blocks will still satisfy requests
//   within 1MB of the oversize cached block size.
//
// With this allocator, allocations and frees should logically be considered
// "usages" of the memory segment associated with streams, just like kernel
// launches. The programmer must insert the proper synchronization if memory
// segments are used from multiple streams.
//
// The library provides a recordStream() function to help insert the correct
// synchronization when allocations are used on multiple streams. This will
// ensure that the block is not reused before each recorded stream completes
```

