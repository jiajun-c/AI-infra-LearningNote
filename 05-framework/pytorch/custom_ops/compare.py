"""
自定义 CUDA 算子 CUDA Graph 兼容性对比 Demo
=============================================
对比 pybind11 vs TORCH_LIBRARY_IMPL 两种自定义算子注册方式
在 torch.compile graph break 和 CUDA Graph capture 上的差异。

用法: python3 compare.py
"""

import os
import time
import torch
import torch.nn as nn

# ============================================================
# Part 0: JIT 编译两个 C++/CUDA 模块
# ============================================================
print("=" * 60)
print("  Part 0: JIT 编译自定义算子模块")
print("=" * 60)

from torch.utils.cpp_extension import load

cur_dir = os.path.dirname(os.path.abspath(__file__))

print("[编译] pybind11 版本...")
pybind_module = load(
    name="pybind_vector_add",
    sources=[os.path.join(cur_dir, "pybind_vector_add.cu")],
    extra_cuda_cflags=["-O3"],
    extra_include_paths=[cur_dir],
    verbose=False,
)

print("[编译] TORCH_LIBRARY_IMPL 版本...")
# TORCH_LIBRARY_IMPL 不使用 PYBIND11_MODULE，没有 PyInit_* 导出函数
# 所以需要设置 is_python_module=False，仅加载共享库注册算子到调度器
torchlib_module = load(
    name="torchlib_vector_add",
    sources=[os.path.join(cur_dir, "torchlib_vector_add.cu")],
    extra_cuda_cflags=["-O3"],
    extra_include_paths=[cur_dir],
    is_python_module=False,
    verbose=False,
)

print("[编译完成]\n")


# ============================================================
# Part 1: Eager 模式正确性验证
# ============================================================
def test_eager_correctness():
    print("=" * 60)
    print("  实验1: Eager 模式正确性验证")
    print("=" * 60)
    print("两种绑定方式在 Eager 模式下都能正确计算\n")

    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    ref = a + b

    out_pybind = pybind_module.vector_add(a, b)
    out_torchlib = torch.ops.custom_ops.vector_add(a, b)

    pybind_ok = torch.allclose(ref, out_pybind)
    torchlib_ok = torch.allclose(ref, out_torchlib)

    print(f"  pybind11         Eager 正确性: {'OK' if pybind_ok else 'FAIL'}")
    print(f"  TORCH_LIBRARY    Eager 正确性: {'OK' if torchlib_ok else 'FAIL'}")
    print()


# ============================================================
# Part 2: torch.compile Graph Break 分析
# ============================================================
def test_graph_breaks():
    print("=" * 60)
    print("  实验2: torch.compile Graph Break 分析")
    print("=" * 60)
    print("pybind11 对 torch.compile 是黑盒 → 产生 graph break")
    print("TORCH_LIBRARY_IMPL 注册到调度器 → 无 graph break\n")

    class ModelWithCustomOp(nn.Module):
        def __init__(self, custom_fn):
            super().__init__()
            self.linear = nn.Linear(1024, 1024)
            self.custom_fn = custom_fn

        def forward(self, x):
            x = self.linear(x)
            residual = x
            x = torch.relu(x)
            x = self.custom_fn(x, residual)  # 自定义 vector_add
            x = torch.relu(x)
            return x

    x = torch.randn(8, 1024, device="cuda")

    # pybind11 版本
    model_pybind = ModelWithCustomOp(pybind_module.vector_add).cuda().eval()
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(model_pybind)(x)
    print(f"  pybind11:       Graph 数量 = {explanation.graph_count}, "
          f"Graph Break 数量 = {explanation.graph_break_count}")
    if explanation.break_reasons:
        for r in explanation.break_reasons:
            print(f"    Break 原因: {r.reason}")

    # TORCH_LIBRARY_IMPL 版本
    model_torchlib = (
        ModelWithCustomOp(torch.ops.custom_ops.vector_add).cuda().eval()
    )
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(model_torchlib)(x)
    print(f"  TORCH_LIBRARY:  Graph 数量 = {explanation.graph_count}, "
          f"Graph Break 数量 = {explanation.graph_break_count}")
    print()


# ============================================================
# Part 3: CUDA Graph 兼容性测试 (核心实验!)
# ============================================================
def test_cuda_graph_compatibility():
    print("=" * 60)
    print("  实验3: CUDA Graph 兼容性 (核心实验!)")
    print("=" * 60)
    print("pybind11 使用默认 stream → kernel 不被 CUDA Graph 捕获")
    print("TORCH_LIBRARY 使用 getCurrentCUDAStream → 正确捕获\n")

    N = 1024 * 1024

    # === 测试 pybind11 + CUDA Graph ===
    print("--- pybind11 + CUDA Graph ---")
    a1 = torch.randn(N, device="cuda")
    b1 = torch.randn(N, device="cuda")

    # Warmup (CUDA Graph 要求先 warmup)
    c1 = pybind_module.vector_add(a1, b1)
    torch.cuda.synchronize()

    # Capture
    graph1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph1):
        c1 = pybind_module.vector_add(a1, b1)

    # 修改输入数据 (CUDA Graph 通过地址绑定，修改原地址数据)
    a1.fill_(1.0)
    b1.fill_(2.0)
    ref1 = a1 + b1  # 期望全是 3.0

    # Replay — pybind11 的 kernel 没被捕获，所以 c1 不会更新
    graph1.replay()
    torch.cuda.synchronize()

    correct1 = torch.allclose(c1, ref1)
    print(f"  Python层报错:   没有! (这就是最危险的地方)")
    print(f"  Replay结果正确: {correct1}")
    print(f"  期望值 (全3.0): {ref1[:5].tolist()}")
    print(f"  实际值:          {c1[:5].tolist()}")
    if not correct1:
        print(f"  !! 结果错误! pybind11 kernel 没有被 CUDA Graph 捕获!")
        print(f"  !! PyTorch Python 层没有任何报错 — 生产环境中极难发现!")

    # === 测试 TORCH_LIBRARY_IMPL + CUDA Graph ===
    print("\n--- TORCH_LIBRARY_IMPL + CUDA Graph ---")
    a2 = torch.randn(N, device="cuda")
    b2 = torch.randn(N, device="cuda")

    # Warmup
    c2 = torch.ops.custom_ops.vector_add(a2, b2)
    torch.cuda.synchronize()

    # Capture
    graph2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph2):
        c2 = torch.ops.custom_ops.vector_add(a2, b2)

    # 修改输入
    a2.fill_(1.0)
    b2.fill_(2.0)
    ref2 = a2 + b2

    # Replay
    graph2.replay()
    torch.cuda.synchronize()

    correct2 = torch.allclose(c2, ref2)
    print(f"  Capture 成功:    是")
    print(f"  Replay结果正确: {correct2}")
    print(f"  期望值 (全3.0): {ref2[:5].tolist()}")
    print(f"  实际值:          {c2[:5].tolist()}")
    print()


# ============================================================
# Part 4: 根因分析 — Stream 不匹配
# ============================================================
def explain_root_cause():
    print("=" * 60)
    print("  实验4: 根因分析 — Stream 不匹配")
    print("=" * 60)
    print("""
正常执行 (Eager):
  PyTorch 默认 stream ──→ pybind11 用 stream 0 ──→ 同一个 stream, 没问题

CUDA Graph Capture:
  PyTorch 切换到 capture stream ──→ pybind11 仍用 stream 0
                                     ↓
                          kernel 在 stream 0 上执行
                          但 graph 只捕获 capture stream 上的操作
                                     ↓
                          kernel 未被捕获! Replay 时不会重放!

TORCH_LIBRARY_IMPL 的修复:
  at::cuda::getCurrentCUDAStream() 返回 capture stream
  → kernel 在 capture stream 上执行 → 被 graph 正确捕获

注意: 即使修复了 stream 问题，pybind11 仍然无法解决:
  • torch.compile graph break (Dynamo 无法 trace)
  • 无 Meta/FakeTensor 实现 (compile 无法推导 shape)
""")


# ============================================================
# Part 5: 性能对比
# ============================================================
def benchmark_performance():
    print("=" * 60)
    print("  实验5: 性能对比 (Eager / torch.compile / CUDA Graph)")
    print("=" * 60)

    N = 1024 * 1024
    NUM_ITERS = 4
    NUM_WARMUP = 10
    NUM_RUNS = 200

    class MultiOpModel(nn.Module):
        """多次调用自定义算子，放大性能差异"""
        def __init__(self, custom_fn):
            super().__init__()
            self.custom_fn = custom_fn

        def forward(self, a, b):
            c = a
            for _ in range(NUM_ITERS):
                c = self.custom_fn(c, b)
                c = torch.relu(c)
            return c

    def measure_time(fn, num_warmup, num_runs):
        """使用 CUDA events 精确计时"""
        for _ in range(num_warmup):
            fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_runs):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / num_runs

    a = torch.randn(N, device="cuda")
    b = torch.randn(N, device="cuda")

    results = {}

    # --- pybind11 ---
    model_pybind = MultiOpModel(pybind_module.vector_add).cuda().eval()

    # Eager
    t = measure_time(lambda: model_pybind(a, b), NUM_WARMUP, NUM_RUNS)
    results[("pybind11", "Eager")] = t

    # torch.compile (有 graph break)
    torch._dynamo.reset()
    compiled_pybind = torch.compile(model_pybind)
    t = measure_time(lambda: compiled_pybind(a, b), NUM_WARMUP, NUM_RUNS)
    results[("pybind11", "compile")] = t

    # CUDA Graph — pybind11 不兼容，跳过
    results[("pybind11", "CUDAGraph")] = None

    # --- TORCH_LIBRARY_IMPL ---
    model_torchlib = MultiOpModel(torch.ops.custom_ops.vector_add).cuda().eval()

    # Eager
    t = measure_time(lambda: model_torchlib(a, b), NUM_WARMUP, NUM_RUNS)
    results[("TORCH_LIB", "Eager")] = t

    # torch.compile (无 graph break)
    torch._dynamo.reset()
    compiled_torchlib = torch.compile(model_torchlib)
    t = measure_time(lambda: compiled_torchlib(a, b), NUM_WARMUP, NUM_RUNS)
    results[("TORCH_LIB", "compile")] = t

    # CUDA Graph
    a_g = a.clone()
    b_g = b.clone()
    # warmup
    c_g = model_torchlib(a_g, b_g)
    torch.cuda.synchronize()
    # capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        c_g = model_torchlib(a_g, b_g)
    t = measure_time(lambda: graph.replay(), NUM_WARMUP, NUM_RUNS)
    results[("TORCH_LIB", "CUDAGraph")] = t

    # 打印结果
    print(f"\n  数据量: {N} 元素, 迭代 {NUM_ITERS} 次, 测量 {NUM_RUNS} 次平均\n")
    print(f"  {'方式':<15} {'Eager':>10} {'compile':>10} {'CUDAGraph':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")

    for method in ["pybind11", "TORCH_LIB"]:
        eager = results[(method, "Eager")]
        compile_t = results[(method, "compile")]
        graph_t = results[(method, "CUDAGraph")]
        graph_str = f"{graph_t:.4f} ms" if graph_t else "N/A(错误)"
        print(f"  {method:<15} {eager:>9.4f}ms {compile_t:>9.4f}ms {graph_str:>10}")
    print()


# ============================================================
# Part 6: 汇总表格
# ============================================================
def print_summary():
    print("=" * 60)
    print("  总结对比")
    print("=" * 60)
    print("""
  特性                    pybind11              TORCH_LIBRARY_IMPL
  ─────────────────────  ───────────────────── ──────────────────────
  Eager 模式              正确                  正确
  torch.compile           有 Graph Break        无 Graph Break
  CUDA Graph Capture      静默失败(不报错!)     正确捕获
  CUDA Graph Replay       结果错误              结果正确
  Python 层报错           不报错(极危险!)       N/A (本身正确)
  Meta/FakeTensor         不支持                支持
  Dispatcher 集成         不集成                完全集成

  建议:
  • 新项目: 始终使用 TORCH_LIBRARY_IMPL 或 torch.library.custom_op
  • 已有 pybind11: 至少使用 at::cuda::getCurrentCUDAStream()
  • 需要 torch.compile: 必须提供 Meta 实现
  • 需要 CUDA Graph: 必须正确处理 stream
""")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print()
    test_eager_correctness()
    test_graph_breaks()
    test_cuda_graph_compatibility()
    explain_root_cause()
    benchmark_performance()
    print_summary()
