import triton
import triton.language as tl
import torch

@triton.jit
def weighted_sum_fwd(
    x_ptr,
    weight_ptr,
    output_ptr,
    x_row_stride,
    x_stride_dim,
    weight_stride_dim,
    output_stride_row,
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    
    x_block_ptr = tl.make_block_ptr(
        x_ptr, 
        shape=(ROWS, D, ),
        strides=(x_row_stride, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D, ),
        strides=(weight_stride_dim, ),
        offsets=(0, ),
        block_shape=(D_TILE_SIZE, ),
        order=(0, ),
    )
    
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS, ),
        strides=(output_stride_row, ),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, ),
        block_shape=(ROWS_TILE_SIZE, ),
        order=(0, ),
    )
    
    ouptut = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        ouptut += tl.sum(row * weight[None, :], axis=1)
        # x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        # weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
    tl.store(output_block_ptr, ouptut, boundary_check=(0,))


def call_weighted_sum(x: torch.Tensor, weight: torch.Tensor, ROWS_TILE=128, D_TILE=512) -> torch.Tensor:
    """
    调用 Triton 加权行求和内核
    参数:
        x:      输入矩阵 [batch, ROWS, D] 或 [ROWS, D]
        weight: 权重向量 [D]
        ROWS_TILE: 行分块大小（建议 64/128/256）
        D_TILE: 列分块大小（建议 256/512/1024）
    """
    # 确保输入在 GPU 且为 float32
    assert x.is_cuda and weight.is_cuda, "输入必须位于GPU"
    assert x.dtype == weight.dtype == torch.float32, "仅支持 float32"
    
    # 处理可能的 batch 维度
    original_shape = x.shape
    if x.dim() == 3:
        x = x.view(-1, x.size(-1))  # 合并 batch 和 ROWS 维度
    ROWS, D = x.shape
    
    # 预分配输出空间 [ROWS,] 或 [batch, ROWS]
    output = torch.empty(ROWS, device='cuda', dtype=torch.float32)
    
    # 计算内存步长（兼容非连续张量）
    x_row_stride = x.stride(0)  # 通常为 D（行优先）
    x_dim_stride = x.stride(1)  # 通常为 1
    weight_stride = weight.stride(0)
    output_stride = output.stride(0)
    
    # 配置执行网格（1D网格，按行分块）
    grid = (triton.cdiv(ROWS, ROWS_TILE), 1, 1)
    
    # 调用 Triton 内核
    weighted_sum_fwd[grid](
        x,        # 输入矩阵指针
        weight,   # 权重向量指针
        output,   # 输出指针
        x_row_stride,        # 行步长
        x_dim_stride,        # 列步长
        weight_stride,       # 权重步长（通常1）
        output_stride,       # 输出步长（通常1）
        ROWS, D,             # 矩阵维度
        ROWS_TILE_SIZE=ROWS_TILE,  # 编译时常量
        D_TILE_SIZE=D_TILE         # 编译时常量
    )
    
    return output.view(original_shape[:-1]) if len(original_shape) == 3 else output

# 验证示例
if __name__ == "__main__":
    # 测试用例配置
    BATCH = 4    # 可选batch维度
    ROWS = 3000   # 测试行数（非分块整数倍）
    D = 8193     # 测试列数（非分块整数倍）
    
    # 生成随机数据（GPU）
    x = torch.randn(BATCH, ROWS, D, device='cuda', dtype=torch.float32)
    weight = torch.randn(D, device='cuda', dtype=torch.float32)
    
    # 调用 Triton 实现
    output_triton = call_weighted_sum(x, weight, ROWS_TILE=128, D_TILE=512)
    
    # 计算 PyTorch 原生结果（验证正确性）
    output_pytorch = (x * weight).sum(dim=-1)
    
    # 对比结果
    max_error = torch.max(torch.abs(output_triton - output_pytorch)).item()
    print(f"最大绝对误差: {max_error:.6f}")
    print("测试通过!" if max_error < 1e-4 else "结果异常!")
    