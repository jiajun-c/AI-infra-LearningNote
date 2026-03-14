# 硬件信息

通过`torch.cuda.current_device()` 可以得到设备对象，然后通过`driver.active.utils.get_device_properties(device)` 可以得到设备属性的信息

```python3
import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = torch.device("cuda:0")

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
print("sm number: ", NUM_SM, "\n register number: ", NUM_REGS, "\n shared memory size: ", SIZE_SMEM, "\n warp size: ", WARP_SIZE)
```

通过这个硬件信息结合我们预编译kernel中的硬件使用情况可以帮助我们更好地设置kernel参数。

```python3
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y
```