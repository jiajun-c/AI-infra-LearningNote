import triton
import triton.language as tl
import torch

@triton.jit
def softmax_tlkernel(
    X,
    Y,
    stride_x,
    stride_y,
    N,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_row_ptr = X + row_idx * stride_x
    y_row_ptr = Y + row_idx * stride_y
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x_ptr = x_row_ptr + mask
    x = tl.load(x_ptr, mask, other=0.0).to(tl.float32)
    input_val = tl.load(x_ptr, mask=mask, other=-float('inf')).to(tl.float32)
    
    # 6. 数值稳定性优化 (Safe Softmax)
    # 先减去最大值，防止 exp 溢出
    max_val = tl.max(input_val, axis=0)
    input_val = input_val - max_val
    
    numerator = tl.exp(input_val)
    denominator = tl.sum(numerator, axis=0)
    
    # 8. 计算最终结果
    y = numerator / denominator
    
    # 9. 写回
    y_ptrs = y_row_ptr + offsets
    tl.store(y_ptrs, y, mask=mask)
    
    
def softmax(x):
    M, N = x.shape
    # Block Size 取大于 N 的最小 2 的幂次
    BLOCK_SIZE = triton.next_power_of_2(N)
    y = torch.empty_like(x)
    
    # 1D grid，每个 program 处理一行
    grid = (M, )
    
    softmax_tlkernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

# --- 测试 ---
if __name__ == "__main__":
    M, N = 4, 10 # 这里的 N 不是 2 的幂次，测试 mask 逻辑
    x = torch.randn(M, N, device='cuda')
    
    # Triton 结果
    y_triton = softmax(x)
    # Torch 结果
    y_torch = torch.softmax(x, dim=1)
    
    print(torch.allclose(y_torch, y_torch))
    print("Max Diff:", (y_triton - y_torch).abs().max())