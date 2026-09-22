"""EP (Expert Parallelism) 的分布式实现 —— 多卡 NCCL 目标版本。

运行：
    torchrun --nproc_per_node=8 10-dist/ep/dist_ep.py
    # 没有多卡时用 gloo + CPU 验证逻辑（语义与 NCCL 一致，只是慢）：
    EP_BACKEND=gloo torchrun --nproc_per_node=4 10-dist/ep/dist_ep.py

与 impl.py 的分工：
    impl.py     单进程模拟，讲清楚【索引搬运】—— 哪个 token 去哪个专家的哪个槽位
    dist_ep.py  真分布式，讲清楚【通信】—— 这些搬运如何变成两次 all_to_all

三个关键点：
    ★1  all_to_all 的梯度 = 把 split sizes 对调后再 all_to_all 一次
    ★2  变长通信必须先交换 metadata，因为 all_to_all_single 要求提前给出 split sizes
    ★3  回程的 split sizes 就是去程的两个 list 对调，不需要重新协商
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ═════════════════════════════════════════════════════════════════════════════
# 通信层
# ═════════════════════════════════════════════════════════════════════════════
def _all_to_all(x: torch.Tensor, out_splits: list, in_splits: list) -> torch.Tensor:
    """裸通信，不带 autograd。

    把 x 的前 in_splits[0] 行发给 rank0、接着 in_splits[1] 行发给 rank1……收回来的
    是「各 rank 发给我的行」按 rank 顺序拼接，第 i 段长 out_splits[i]。

    输出必须按 out_splits 分配而不是 empty_like(x) —— 收发长度通常不等。
    """
    out = x.new_empty((sum(out_splits), *x.shape[1:]))
    dist.all_to_all_single(out, x.contiguous(),
                           output_split_sizes=list(out_splits),
                           input_split_sizes=list(in_splits))
    return out


class _AllToAll(torch.autograd.Function):
    """★1 all_to_all 的 autograd 封装。

    all_to_all 是线性算子，它的转置就是它自己 —— 只要把输入/输出的切分方式对调：

        forward :  x [sum(in)]  →  g [sum(out)]
        backward:  g [sum(out)] →  dx [sum(in)]   即 _all_to_all(g, in, out)

    有了这一层，dispatch/compute/combine 里的 index_select / cat / matmul 全交给
    PyTorch 自动求导，不必手写 expert 的 backward。
    """

    @staticmethod
    def forward(ctx, x, out_splits, in_splits):
        # split sizes 是 Python list，不是 tensor —— 只能挂属性。
        # save_for_backward 只接受 Tensor，这正是它的设计边界。
        ctx.out_splits, ctx.in_splits = out_splits, in_splits
        return _all_to_all(x, out_splits, in_splits)

    @staticmethod
    def backward(ctx, g):
        return _all_to_all(g, ctx.in_splits, ctx.out_splits), None, None


def _all_gather_counts(send_counts: torch.Tensor) -> torch.Tensor:
    """★2 交换 metadata：send_counts[i] = 我要发给 rank i 的行数，返回「各 rank 发给我的行数」。

    必须交换的原因：all_to_all_single 要求提前给出 split sizes，但「我会收到多少」
    只有等所有 rank 都做完 gating 才知道。只有 P 个整数，开销可忽略。
    """
    P = dist.get_world_size()
    buf = [torch.empty_like(send_counts) for _ in range(P)]
    dist.all_gather(buf, send_counts.contiguous())
    return torch.stack(buf)[:, dist.get_rank()].contiguous()   # [src, dst] 取第 rank 列


def _inverse_perm(p: torch.Tensor) -> torch.Tensor:
    """p 是置换，返回 q 满足 q[p[i]] = i（用于把重排过的数据还原回去）。"""
    q = torch.empty_like(p)
    q[p] = torch.arange(p.numel(), device=p.device)
    return q


# ═════════════════════════════════════════════════════════════════════════════
# 模型组件
# ═════════════════════════════════════════════════════════════════════════════
class Expert(nn.Module):
    """单个 FFN 专家：d_model -> d_ff -> d_model。

    参数按 [in, out] 存放，全程不需要 .t()。（nn.Linear 内部是 [out, in]，用
    F.linear 也能免转置；这里用裸 matmul 是因为真实 EP 常要对专家权重按 index 切片。）
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.w2 = nn.Parameter(torch.empty(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x @ self.w1) @ self.w2


class Gate(nn.Module):
    """Router。各 rank 的权重必须**完全相同** —— 这是 gating 不需要通信的原因，
    也是 globals 能对齐的前提（真实做法见 DistEP.sync_gate）。"""

    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.w = nn.Parameter(torch.empty(d_model, num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x @ self.w, dim=-1)


# ═════════════════════════════════════════════════════════════════════════════
# EP 主体
# ═════════════════════════════════════════════════════════════════════════════
class DistEP(nn.Module):
    """每个 rank 只持有 EPR = N/P 个专家。

    x [T, d]
      ├─ gate → top_k → (全局专家 id, 门控权重)        ← 无通信
      ├─ 按 dest = e // EPR 分桶
      ├─ all_gather(send_counts)                       ← 通信 0：P 个整数
      ├─ all_to_all × 2  (hidden, expert_id)           ← 通信 1：DISPATCH
      ├─ 本地按专家分组 → 逐专家 FFN                     ← 无通信
      ├─ all_to_all × 1  (hidden)                      ← 通信 2：COMBINE
      └─ 回到源 rank 后乘门控权重，把 K 个输出相加       ← 无通信
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int,
                 seed: int = 0):
        super().__init__()
        P = dist.get_world_size()
        assert num_experts % P == 0, "专家数必须能被 EP 度整除"
        self.world_size, self.num_experts, self.top_k = P, num_experts, top_k
        self.experts_per_rank, self.rank = num_experts // P, dist.get_rank()

        self.gate = Gate(d_model, num_experts)
        # 必须用 nn.ModuleList —— 普通 list 里的参数不会注册进 parameters()
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(self.experts_per_rank)])

        # gate 各 rank 用同一种子 → 相同；再由 rank0 广播兜底（这才是真实做法）
        _init(self.gate, seed)
        # 专家各 rank 必须【互不相同】—— 它们是被切分开的不同专家，不是副本。
        # 若各 rank 用同一种子各自初始化，所有专家会退化成同一个函数 E，此时
        #     out = Σ_k w_k·E(x) = E(x)·Σ_k w_k = E(x)
        # 门控权重被完全约掉，d(loss)/dw_gate 解析上恒为 0 —— 实测表现为 ~1e-7 的
        # fp32 舍入噪声而非 0，很容易被误判成梯度 bug。所以用 rank 相关种子错开。
        _init(self.experts, seed + 1000 * (self.rank + 1))
        self.sync_gate()

    def sync_gate(self) -> None:
        for p in self.gate.parameters():
            dist.broadcast(p.data, src=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        P, EPR, K = self.world_size, self.experts_per_rank, self.top_k
        T, d = x.shape

        # ── 0. gating（无通信）：gate 全 rank 相同 → 各卡路由决策一致 ──
        top_w, top_idx = torch.topk(self.gate(x), K, dim=-1)     # [T, K]
        top_w = top_w / top_w.sum(-1, keepdim=True)              # 在 top-K 上重新归一化

        # ── 1. 展平成 (token, expert) 对，按目标 rank 分桶 ────────────────
        # 第 j 行 ⇔ 「第 j//K 个 token 的第 j%K 个选择」，x.repeat_interleave 与
        # top_idx.reshape(-1) 的行序天然对齐
        pair_h = x.repeat_interleave(K, dim=0)                   # [T*K, d]
        pair_e = top_idx.reshape(-1)                             # [T*K] 全局专家 id
        dest = pair_e // EPR                                     # [T*K] 目标 rank

        # stable argsort 让同一 dest 的行连续且 dest 递增 → 分组边界就是累加和
        order = torch.argsort(dest, stable=True)
        send_h, send_e = pair_h[order], pair_e[order]
        send_counts = torch.bincount(dest, minlength=P)          # [P]

        # ── 2. DISPATCH（通信 1）────────────────────────────────────────
        si = send_counts.tolist()                                # 我发出去的切分
        so = _all_gather_counts(send_counts).tolist()             # 我会收到的切分
        recv_h = _AllToAll.apply(send_h, so, si)                 # 浮点 → 走 autograd
        recv_e = _all_to_all(send_e, so, si)                     # int64 索引 → 无需梯度

        # 收到后只应包含本 rank 负责的专家 —— 对发送端路由的断言
        assert recv_e.numel() == 0 or bool((recv_e // EPR == self.rank).all()), \
            "dispatch 路由错位"

        # ── 3. COMPUTE（无通信）：按本地专家分组做 GEMM ─────────────────
        # 收到时数据按【来源 rank】排列，跑分组 GEMM 得先按【本地专家】重排
        local_e = recv_e % EPR
        perm = torch.argsort(local_e, stable=True)
        counts = torch.bincount(local_e, minlength=EPR)
        # 某个专家可能一个 token 都没收到 → 有 0 长度的 chunk。matmul 对 [0,d]
        # 合法，所以不需要 if 分支。
        ys = [self.experts[i](c)
              for i, c in enumerate(torch.split(recv_h[perm], counts.tolist()))]
        out_local = torch.cat(ys)[_inverse_perm(perm)]           # 还原成 recv 的顺序

        # ── 4. COMBINE（通信 2）─────────────────────────────────────────
        # ★3 回程切分就是去程两个 list 对调：我从 rank i 收了多少行就还回去多少行，
        #    不需要重新协商 metadata。
        out_pairs = _AllToAll.apply(out_local, si, so)
        # 门控权重在源 rank 本地乘 —— 所以 w 不必跟着搬过去（省掉一次集合通信）
        return (out_pairs[_inverse_perm(order)].view(T, K, d)
                * top_w.unsqueeze(-1)).sum(dim=1)


def _init(module: nn.Module, seed: int, std: float = 0.02) -> None:
    """用独立 generator 初始化，不污染全局 RNG。"""
    g = torch.Generator().manual_seed(seed)
    for p in module.parameters():
        p.data.normal_(0.0, std, generator=g)


# ═════════════════════════════════════════════════════════════════════════════
# 正确性验证：和「不切分的稠密实现」对拍
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def verify_against_dense(model: DistEP, x: torch.Tensor) -> float:
    """返回 EP 输出与单卡稠密 MoE 的最大绝对误差。

    稠密参考需要完整的 N 个专家，所以把所有 rank 的专家参数拼起来 —— 仅用于验证。
    真实训练中没人这么干，那正是 EP 要避免的。
    """
    P = dist.get_world_size()
    N, K = model.num_experts, model.top_k

    def gather(t):                                   # [P, *t.shape]
        buf = [torch.empty_like(t) for _ in range(P)]
        dist.all_gather(buf, t.contiguous())
        return torch.stack(buf)

    # 局部第 j 个专家在各 rank 拼起来是全局专家 j, EPR+j, 2EPR+j…
    # stack(dim=1) → [P, EPR, …] → flatten 后索引正好是 e = p*EPR + j
    W1 = torch.stack([gather(e.w1) for e in model.experts], 1).flatten(0, 1)
    W2 = torch.stack([gather(e.w2) for e in model.experts], 1).flatten(0, 1)

    top_w, top_idx = torch.topk(model.gate(x), K, dim=-1)
    top_w = top_w / top_w.sum(-1, keepdim=True)
    y = torch.zeros_like(x)
    for k in range(K):
        for e in range(N):
            m = top_idx[:, k] == e                   # 哪些 token 的第 k 个选择是专家 e
            if m.any():
                y[m] += top_w[m, k:k + 1] * (F.gelu(x[m] @ W1[e]) @ W2[e])

    out_all = [torch.empty_like(x) for _ in range(P)]
    dist.all_gather(out_all, model(x).contiguous())
    return (out_all[dist.get_rank()] - y).abs().max().item()


# ═════════════════════════════════════════════════════════════════════════════
# 入口
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    backend = os.environ.get("EP_BACKEND",
                             "nccl" if torch.cuda.is_available() else "gloo")
    dist.init_process_group(backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")          # gloo 只在 CPU 张量上工作

    model = DistEP(64, 128, 8, 2, seed=0).to(device)
    gx = torch.Generator().manual_seed(1234)  # 固定输入，便于跨 world_size 对比
    x = torch.randn(16, 64, device=device, generator=gx, requires_grad=True)

    out = model(x)
    out.sum().backward()

    err = verify_against_dense(model, x.detach())
    if rank == 0:
        print(f"backend={backend}  world_size={world}  "
              f"每 rank 专家数={model.experts_per_rank}/{model.num_experts}")
        print(f"与稠密参考最大误差: {err:.3e}   （fp32 下应在 1e-5 量级）")
        print(f"gate 梯度范数     : {model.gate.w.grad.norm():.3e}   （退化时会是 ~1e-7）")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
