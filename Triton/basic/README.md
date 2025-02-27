# Triton

## 1. 安装

当前尽量使用python3.9的版本，DEVICE使用 `DEVICE = torch.device("cuda:0")`

## 2. 基础使用

在Triton中，任务是会被划分到多个线程上，对于每个线程通过`program_id` 来进行识别

`pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.` 

获取块的起始位置和结束位置，并和总长度进行比较来得到计算的Mask

```python3
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
```

进行数据load，通过 `tl.load` 加载数据，然后通过 `tl.store` 去存储计算的结果

```python3
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
```

启动函数，在Trtion中，grid层面的参数传递方式和cuda类似，对于BLOCK层面的信息我们仅需要在函数参数中进行指定即可

```python3
def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel() # numel 返回元素的总数
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output
```