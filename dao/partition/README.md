# 任务划分

在算子的任务划分中往往会存在两种方式

- 以维度为中心，比如按照batch维度进行并行
- 以硬件为中心，比如根据SM的数量进行划分
- 折中的方案，split-k，通过对某一维度进行切分产生更高的并行效率

以RMSNorm的反向传播为例，采用的方案就是按sm的数量进行划分，然后不断遍历需要并行的维度，

```cpp
    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
```