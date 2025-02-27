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