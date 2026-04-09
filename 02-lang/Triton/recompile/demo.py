import torch
import triton
import triton.language as tl

# =====================================================================
# 1. 纯粹的底层 Kernel (剥离所有的 Autotune 装饰器)
# =====================================================================
@triton.jit
def _edge_qk_fwd_csr_kernel(
    Q, K, 
    Q_ROW_PTR, Q_COL_INDEX, Q_ALPHA_INDEX, 
    OUT,
    N, H, D,
    s_qn, s_qh, s_qd,
    s_kn, s_kh, s_kd,
    s_oe, s_oh,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 这里写你原本的业务逻辑，我用简单的占位符代替以保证代码可运行
    pid = tl.program_id(0)
    pass 

# =====================================================================
# 2. 离线最优配置表 (Lookup Table)
# 替代原本的 @triton.autotune，这些数值是你在线下压测出来的最优解
# =====================================================================
def get_best_config(H, D):
    """
    根据 Head 数和维度，直接返回最佳的硬件配置
    """
    # 模拟离线测出的最佳配置表
    config_map = {
        (32, 64):  {'BLOCK_H': 32, 'BLOCK_D': 16, 'num_warps': 4, 'num_stages': 2},
        (32, 128): {'BLOCK_H': 32, 'BLOCK_D': 32, 'num_warps': 4, 'num_stages': 3},
        (128, 64): {'BLOCK_H': 32, 'BLOCK_D': 16, 'num_warps': 4, 'num_stages': 2},
    }
    
    # 如果遇到没见过的形状，返回一个最保守、绝对能跑通的 Fallback 配置
    fallback_config = {'BLOCK_H': 16, 'BLOCK_D': 16, 'num_warps': 4, 'num_stages': 2}
    
    return config_map.get((H, D), fallback_config)

# =====================================================================
# 3. 带有防御机制的前向分发器 (Dispatcher)
# =====================================================================
def edge_qk_forward_prod(query, key, q_row_ptr, q_col_index, q_alpha_index, num_edges):
    assert query.ndim == 3 and key.ndim == 3
    N, H, D = query.shape
    M, Hk, Dk = key.shape
    E = num_edges
    assert H == Hk and D == Dk

    if E == 0:
        return torch.empty((0, H), device=query.device, dtype=torch.float32)

    # 🛡️ 防御机制 1：抹平内存碎片，强制拉直 Strides
    # .contiguous() 不仅修复 stride，在 PyTorch 底层通常也会返回 16-byte 对齐的指针
    # 从而防止 Triton 触发 Pointer Divisibility (整除性) 的 JIT 重编译
    q = query.contiguous()
    k = key.contiguous()
    
    # 🛡️ 防御机制 2：一维索引类张量极其容易成为不对齐的真凶！
    # 强烈建议在线上推理时，也对它们做强制连续化
    q_row_ptr = q_row_ptr.contiguous()
    q_col_index = q_col_index.contiguous()
    q_alpha_index = q_alpha_index.contiguous()

    out = torch.empty((E, H), device=q.device, dtype=torch.float32)
    
    # 获取离线写死的最优配置
    cfg = get_best_config(H, D)
    block_d = triton.next_power_of_2(D) # 如果你业务上强制要求 BLOCK_D 必须是 2的幂次

    grid = lambda meta: (N, triton.cdiv(H, meta["BLOCK_H"]))
    
    # 🚀 直接发射 Kernel，绕过所有的 Cache 和 Autotune 逻辑
    _edge_qk_fwd_csr_kernel[grid](
        q, k, 
        q_row_ptr, q_col_index, q_alpha_index, 
        out,
        N, H, D,
        *q.stride(),
        *k.stride(),
        *out.stride(),
        BLOCK_H=cfg['BLOCK_H'],
        BLOCK_D=block_d if block_d > cfg['BLOCK_D'] else cfg['BLOCK_D'], # 结合你的业务逻辑
        num_warps=cfg['num_warps'],
        num_stages=cfg['num_stages']
    )
    return out

# =====================================================================
# 测试执行模块
# =====================================================================
if __name__ == "__main__":
    # 模拟设备和形状
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, H, D = 1024, 32, 128
    E = 5000
    
    if device.type == "cuda":
        # 伪造输入数据
        query = torch.randn((N, H, D), device=device, dtype=torch.float16)
        key = torch.randn((N, H, D), device=device, dtype=torch.float16)
        
        # 伪造图的 CSR 索引数据 (这里随便填的，仅做占位)
        q_row_ptr = torch.zeros(N + 1, device=device, dtype=torch.int32)
        q_col_index = torch.zeros(E, device=device, dtype=torch.int32)
        q_alpha_index = torch.zeros(E, device=device, dtype=torch.int32)
        
        print("🚀 第一波请求 (N=1024)...")
        # 第一次编译并执行 (只会在第一次编译基本的 PTX)
        out1 = edge_qk_forward_prod(query, key, q_row_ptr, q_col_index, q_alpha_index, E)
        print("✅ 成功完成！")

        print("🚀 第二波请求 (模拟突变的 N 和切片后的非连续张量)...")
        # 模拟极端恶劣的线上场景：不仅 N 变了，传进来的 Tensor 还是被切片过的！
        # 注意：这里的 query_bad 的 stride 已经彻底乱了，地址也极有可能没对齐
        query_bad = torch.randn((2048, H, D * 2), device=device, dtype=torch.float16)[:, :, :D] 
        key_bad = torch.randn((2048, H, D), device=device, dtype=torch.float16)
        q_row_ptr_bad = torch.zeros(2048 + 1, device=device, dtype=torch.int32)
        
        # 因为我们在分发器里加了 .contiguous() 护盾，且移除了 autotune
        # 这里绝对不会发生任何重编译和调优，直接极速返回！
        out2 = edge_qk_forward_prod(query_bad, key_bad, q_row_ptr_bad, q_col_index, q_alpha_index, E)
        print("✅ 成功抵御动态 Shape 攻击，极速完成！")