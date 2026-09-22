"""
EP (Expert Parallelism) 最小实现
================================

单进程模拟 P 个 rank 的 EP，把 dispatch / compute / combine 三阶段的
索引搬运全部显式写出来，并与「不做 EP 的稠密参考」逐步对拍。

为什么要单进程模拟：
  真实 EP 的正确性验证需要多卡，但 EP 的全部逻辑难点在【索引】——
  哪个 token 该去哪个 rank 的哪个专家、算完怎么按原位置加权求和回来。
  这部分用单进程 + 显式索引写清楚，换成真实 all_to_all 只是替换搬运层。

真实分布式对应关系：
    dispatch  → dist.all_to_all_single(...)  /  DeepEP buffer.dispatch()
    compute   → 本 rank 的专家 FFN
    combine   → dist.all_to_all_single(...)  /  DeepEP buffer.combine()

运行：
    python impl.py
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MoEConfig:
    d_model: int = 64
    d_ff: int = 128
    num_experts: int = 8          # N：专家总数
    top_k: int = 2                # K：每个 token 激活几个专家
    num_ranks: int = 4            # P：EP 并行度
    tokens_per_rank: int = 16     # T：每个 rank 持有的 token 数
    capacity_factor: float = 1.25  # CF
    dtype_bytes: int = 2          # bf16

    @property
    def experts_per_rank(self) -> int:
        assert self.num_experts % self.num_ranks == 0, "专家数必须能被 EP 度整除"
        return self.num_experts // self.num_ranks

    @property
    def capacity(self) -> int:
        """每个专家的容量上限。

        注意基数：每个专家收到的是【全局】token，不是单个 rank 的 token。
        因为 EP 下所有 rank 的 token 都会路由到所有专家。

        Switch Transformer (top-1):  capacity = CF * T_global / N
        通用 top-K:                  capacity = CF * T_global * K / N
        """
        t_global = self.tokens_per_rank * self.num_ranks
        return int(self.capacity_factor * t_global * self.top_k / self.num_experts)


# ─────────────────────────────────────────────────────────────────────────────
# 模块
# ─────────────────────────────────────────────────────────────────────────────
class Expert(nn.Module):
    """一个 FFN 专家：d -> d_ff -> d"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(d_model, d_ff) * 0.02)
        self.w2 = nn.Parameter(torch.randn(d_ff, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x @ self.w1) @ self.w2


class Gate(nn.Module):
    """Router：x -> [T, N] 的门控概率"""

    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.wg = nn.Parameter(torch.randn(num_experts, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x @ self.wg.t(), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 稠密参考：不做 EP，不切分，直接算真值
# ─────────────────────────────────────────────────────────────────────────────
def dense_forward(x, gate, experts, cfg):
    """真值实现：y = Σ_k p̃_{e_k} · E_{e_k}(x)

    返回 (out, probs, top_idx, top_w)，后三者供 EP 路径复用，
    保证两条路径面对【完全相同】的路由决策，只比较搬运是否正确。
    """
    probs = gate(x)                                  # [T, N]
    top_w, top_idx = torch.topk(probs, cfg.top_k, dim=-1)   # [T, K]
    top_w = top_w / top_w.sum(-1, keepdim=True)              # 在 top-K 上重新归一化

    out = torch.zeros_like(x)
    for k in range(cfg.top_k):
        for e in range(cfg.num_experts):
            mask = top_idx[:, k] == e                # 哪些 token 的第 k 个选择是专家 e
            if mask.any():
                out[mask] += top_w[mask, k:k + 1] * experts[e](x[mask])
    return out, probs, top_idx, top_w


# ─────────────────────────────────────────────────────────────────────────────
# EP 路径：显式的 dispatch → compute → combine
# ─────────────────────────────────────────────────────────────────────────────
def ep_forward(x_list, gate, experts, cfg):
    """模拟 P 个 rank 的 EP 前向。

    x_list[r] : [T, d]  —— rank r 本地持有的 token
    experts   : 全部 N 个专家（模拟时都在一个进程里，但【归属】按 rank 划分）

    返回 out_list[r] : [T, d]，应等于稠密参考中属于 rank r 的那一段。
    """
    P, N, K = cfg.num_ranks, cfg.num_experts, cfg.top_k
    EPR = cfg.experts_per_rank
    T, d = x_list[0].shape

    # ── 阶段 0：本地 gating（每个 rank 独立算，无通信）─────────────────────
    # 真实实现里这一步也在本地，因为 gate 权重每张卡都有一份完整的。
    # gate 是确定性的（softmax of linear），所以这里算出的路由与稠密参考
    # 完全一致 —— 两条路径面对相同的路由决策，差异只可能来自【搬运】。
    top_idx, top_w = [], []
    for r in range(P):
        probs_r = gate(x_list[r])                                # [T, N]
        w_r, idx_r = torch.topk(probs_r, K, dim=-1)              # [T, K]
        top_w.append(w_r / w_r.sum(-1, keepdim=True))            # top-K 上重新归一
        top_idx.append(idx_r)

    # ── 阶段 1：Dispatch ────────────────────────────────────────────────
    # 目标：把 (src_rank, token_i) 的第 k 个专家选择，搬到
    #       dest_rank = expert_id // EPR 的本地专家槽位上。
    #
    # 真实实现是 all_to_all：每个 rank 把发给其他 rank 的数据打包成一个
    # send buffer，一次通信全部交换。这里用 Python 列表模拟这个 buffer。
    recv = [[[] for _ in range(EPR)] for _ in range(P)]   # recv[rank][local_expert] = [(src, tok_i, w, x_row)]
    for src in range(P):
        for k in range(K):
            for tok_i in range(T):
                e = int(top_idx[src][tok_i, k])
                dest = e // EPR
                local_e = e % EPR
                recv[dest][local_e].append((src, tok_i, float(top_w[src][tok_i, k]), x_list[src][tok_i]))

    # ── 阶段 2：Compute ────────────────────────────────────────────────
    # 每个 rank 只跑自己那几个专家。注意：这一步是【按专家分组】的
    # "grouped GEMM" —— 不同专家收到的 token 数不同，实际实现用
    # 变长分组矩阵乘（这就是 DeepEP 要解决 expert_alignment 的原因）。
    results = [[[] for _ in range(EPR)] for _ in range(P)]
    for rank in range(P):
        for local_e in range(EPR):
            group = recv[rank][local_e]
            if not group:
                continue
            e = rank * EPR + local_e
            rows = torch.stack([g[3] for g in group])          # [n_tok, d]
            y = experts[e](rows) * torch.tensor(                 # 乘门控权重
                [[g[2]] for g in group], dtype=rows.dtype)
            for j, (src, tok_i, _, _) in enumerate(group):
                results[rank][local_e].append((src, tok_i, y[j]))

    # ── 阶段 3：Combine ────────────────────────────────────────────────
    # 把结果送回原 rank，并按 token 汇总（同一 token 的 K 个专家输出相加）。
    # 真实实现是第二次 all_to_all；这里同样用列表模拟。
    out = [torch.zeros_like(x_list[r]) for r in range(P)]
    for rank in range(P):
        for local_e in range(EPR):
            for src, tok_i, y_row in results[rank][local_e]:
                out[src][tok_i] += y_row

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 容量因子 / token 丢弃
# ─────────────────────────────────────────────────────────────────────────────
def apply_capacity(top_idx, top_w, cfg):
    """按容量上限截断路由，返回 (新 top_idx, 新 top_w, 被丢弃数)。

    语义：每个专家最多收 capacity 个 token，超出的【按先到先得】丢弃，
    该 token 在这个专家位置的权重归零（由 residual 连接兜底）。
    """
    P, N, K, T = cfg.num_ranks, cfg.num_experts, cfg.top_k, cfg.tokens_per_rank
    new_idx = [t.clone() for t in top_idx]
    new_w = [t.clone() for t in top_w]
    count = torch.zeros(N, dtype=torch.long)     # 每个专家已接收数
    dropped = 0

    # 按 (src_rank, token, k) 的到达顺序扫描
    for src in range(P):
        for tok_i in range(T):
            for k in range(K):
                e = int(top_idx[src][tok_i, k])
                if count[e] < cfg.capacity:
                    count[e] += 1
                else:
                    new_w[src][tok_i, k] = 0.0    # 丢弃：权重归零
                    dropped += 1
    return new_idx, new_w, dropped


def aux_loss(probs, top_idx, cfg):
    """L_aux = alpha * N * Σ_i f_i * P_i

    f_i: 专家 i 实际处理的 token 比例
    P_i: router 分配给专家 i 的平均概率
    """
    N = cfg.num_experts
    # 归一化基数是【总指派数】T*P*K，不是 token 数 T*P。
    # top-K 下一个 token 产生 K 次指派，所以 Σf_i 应该 = 1（f_i 是"指派的占比"）。
    total_assignments = probs[0].shape[0] * cfg.num_ranks * cfg.top_k

    f = torch.zeros(N)
    for src in range(cfg.num_ranks):
        for k in range(cfg.top_k):
            for e in range(N):
                f[e] += (top_idx[src][:, k] == e).sum().item()
    f = f / total_assignments                          # 实际占比

    P_i = torch.stack(probs).mean(dim=0).mean(dim=0)   # 平均概率

    alpha = 0.01
    return alpha * N * (f * P_i).sum(), f, P_i


# ─────────────────────────────────────────────────────────────────────────────
# 通信量账本
# ─────────────────────────────────────────────────────────────────────────────
def comm_accounting(cfg, f_actual=None):
    d, b = cfg.d_model, cfg.dtype_bytes
    T, K = cfg.tokens_per_rank, cfg.top_k

    print("  通信量账本（每 rank，每层，单向）")
    print(f"    理论 dispatch : T*K*d*b = {T}*{K}*{d}*{b:<3} = {T * K * d * b / 1024:>8.2f} KiB")
    print(f"    理论 combine  : 同上                       = {T * K * d * b / 1024:>8.2f} KiB")
    print(f"    合计          : 2*T*K*d*b                  = {2 * T * K * d * b / 1024:>8.2f} KiB")

    if f_actual is not None:
        # 完美均衡时每个专家收 T*K/N 个 token。用实际分布算真实下界：
        # 一个 rank 要接收的是【所有落在它专家上的 token】，与均衡度有关。
        n_experts_here = cfg.experts_per_rank
        ideal = T * K / cfg.num_experts * n_experts_here
        worst = T * K
        print(f"    完美均衡时每 rank 实收 ≈ {ideal:.1f} 个 token-copy")
        print(f"    最坏情况（全落一个 rank）= {worst} 个 token-copy  "
              f"→ 放大 {worst / ideal:.1f}x")
        print(f"    ★ padding 到最长 rank 的浪费 = 就是上面这个放大倍数")


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(0)
    torch.set_grad_enabled(False)     # 纯前向演示，不需要建图
    cfg = MoEConfig()
    P, N, K = cfg.num_ranks, cfg.num_experts, cfg.top_k

    print("=" * 72)
    print(f"EP 配置: d_model={cfg.d_model} d_ff={cfg.d_ff} N={N} K={K} "
          f"P={P} T={cfg.tokens_per_rank}")
    print(f"每 rank 专家数 = {cfg.experts_per_rank} | 容量上界 = {cfg.capacity} token/专家")
    print("=" * 72)

    gate = Gate(cfg.d_model, N)
    experts = nn.ModuleList([Expert(cfg.d_model, cfg.d_ff) for _ in range(N)])

    # 每个 rank 的本地 token
    x_list = [torch.randn(cfg.tokens_per_rank, cfg.d_model) for _ in range(P)]

    # ── 1. 稠密参考（真值）─────────────────────────────────────────────
    print("\n[1] 稠密参考（不切分，每卡都有全部专家）")
    ref_out, probs, top_idx, top_w = [], [], [], []
    for r in range(P):
        o, p, idx, w = dense_forward(x_list[r], gate, experts, cfg)
        ref_out.append(o); probs.append(p); top_idx.append(idx); top_w.append(w)
    print(f"    输出 shape = {tuple(ref_out[0].shape)}")

    # ── 2. EP 路径 ─────────────────────────────────────────────────────
    print("\n[2] EP 路径（dispatch → compute → combine）")
    ep_out = ep_forward(x_list, gate, experts, cfg)

    max_err = max((ep_out[r] - ref_out[r]).abs().max().item() for r in range(P))
    print(f"    与稠密参考的最大误差 = {max_err:.3e}")
    assert max_err < 1e-6, "EP 路径与稠密参考不一致！"
    print("    ✓ 一致")

    # ── 3. 通信量账本 ──────────────────────────────────────────────────
    print("\n[3] 通信量账本")
    comm_accounting(cfg)

    # ── 4. 负载均衡 ────────────────────────────────────────────────────
    print("\n[4] 负载均衡")
    loss, f, P_i = aux_loss(probs, top_idx, cfg)
    print(f"    L_aux = {loss.item():.6f}")
    print(f"    各专家实际占比 f_i = {[f'{v:.3f}' for v in f.tolist()]}")
    print(f"    各专家平均概率 P_i = {[f'{v:.3f}' for v in P_i.tolist()]}")
    print(f"    （理想值 f_i = P_i = 1/N = {1/N:.3f}）")
    print(f"    最忙/最闲 = {f.max().item() / max(f.min().item(), 1e-9):.1f}x"
          f"  ← 这个比值决定了 all-to-all 要 padding 多少")

    # ── 5. 容量因子与丢弃 ──────────────────────────────────────────────
    print("\n[5] 容量因子与 token 丢弃")
    for cf in (1.0, 1.25, 2.0, 100.0):
        cfg.capacity_factor = cf
        _, _, dropped = apply_capacity(top_idx, top_w, cfg)
        total = P * cfg.tokens_per_rank * K
        print(f"    CF={cf:<6} 容量={cfg.capacity:<3} 丢弃 {dropped:>3}/{total} "
              f"({dropped / total * 100:>5.1f}%)")

    print("\n" + "=" * 72)
    print("核心结论：EP 把【参数量】和【计算量】解耦：")
    print(f"  稠密 FFN 每 token FLOPs ∝ N = {N}")
    print(f"  MoE top-K  每 token FLOPs ∝ K = {K}   → 省 {N / K:.1f}x 计算")
    print(f"  代价：每层 2 次 all-to-all，通信量 ∝ K*d = {K}*{cfg.d_model}")
    print("=" * 72)


if __name__ == "__main__":
    main()
