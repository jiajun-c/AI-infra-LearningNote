# torch _SymmetricMemory 实现

在torch中当我们调用了 `symm_mem.rendezvous` 将会返回一个 _SymmetricMemory 的对象，

下面的代码主要可以被分为四个部分
- 内存管理
    - set_group_info: 底层环境配置
    - empty_stride_p2p: 申请一块P2P访问的显存
    - has_multicast_support: 检查是否支持硬件多播
- 建立连接
    - rendezvous：建立虚拟地址的映射
    - get_buffer: 获取远端的buffer
- 信号同步
- 底层指针

```python
class _SymmetricMemory:
    @staticmethod
    def set_group_info(
        group_name: str,
        rank: int,
        world_size: int,
        store: Store,
    ) -> None: ...
    @staticmethod
    def empty_strided_p2p(
        size: torch.types._size,
        stride: torch.types._size,
        dtype: torch.dtype,
        device: torch.device,
        group_name: str | None = None,
        alloc_id: int | None = None,
    ) -> torch.Tensor: ...
    @staticmethod
    def has_multicast_support(
        device_type: DeviceType,
        device_idx: int,
    ) -> bool: ...
    @property
    def rank(self) -> int: ...
    @property
    def world_size(self) -> int: ...
    @staticmethod
    def rendezvous(
        tensor: torch.Tensor, group_name: str | None = None
    ) -> _SymmetricMemory: ...
    def get_buffer(
        self,
        rank: int,
        sizes: torch.types._size,
        dtype: torch.dtype,
        storage_offset: int | None = 0,
    ) -> torch.Tensor: ...
    def get_signal_pad(
        self,
        rank: int,
        sizes: torch.types._size = [],
        dtype: torch.dtype | None = None,
        storage_offset: int | None = 0,
    ) -> torch.Tensor: ...
    def barrier(self, channel: int = 0, timeout_ms: int = 0) -> None: ...
    def put_signal(
        self,
        dst_rank: int,
        channel: int = 0,
        timeout_ms: int = 0,
    ) -> None: ...
    def wait_signal(
        self,
        src_rank: int,
        channel: int = 0,
        timeout_ms: int = 0,
    ) -> None: ...
    @staticmethod
    def memset32(
        tensor: torch.Tensor, offset: int, val: int, count: int = 1
    ) -> torch.Tensor: ...
    @staticmethod
    def stream_write_value32(
        tensor: torch.Tensor, offset: int, val: int
    ) -> torch.Tensor: ...
    @property
    def buffer_ptrs(self) -> list[int]: ...
    @property
    def buffer_ptrs_dev(self) -> int: ...
    @property
    def signal_pad_ptrs(self) -> list[int]: ...
    @property
    def signal_pad_ptrs_dev(self) -> int: ...
    @property
    def multicast_ptr(self) -> int: ...
    @property
    def buffer_size(self) -> int: ...
    @property
    def signal_pad_size(self) -> int: ...
```