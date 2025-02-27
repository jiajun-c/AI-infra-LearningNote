import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")

# 对于一个二维的数组，将其划分为若干个行块，每个行块内再使用并行

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0) # 一共有多少个线程
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offset = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offset
        mask = col_offset < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator/denominator
        
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offset
        tl.store(output_ptrs, softmax_output, mask=mask)
        