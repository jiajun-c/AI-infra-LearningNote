import os
import torch
import torch.distributed as dist
import argparse
from torch.profiler import profile, record_function, ProfilerActivity

def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # [极其关键的硬件补丁]
    # 如果只有 4 个空闲 SM，强行压制 NCCL 通道数，防止硬件死锁
    os.environ["NCCL_MAX_NCHANNELS"] = "4"

    dist.init_process_group(backend="nccl", timeout=torch.distributed.Duration(seconds=180))

    return local_rank

def cleanup():
    dist.destroy_process_group()

def run_profile_graphs(local_rank, sm_reserve=4, use_profiler=False, profile_dir="./profile_logs", output_json="graph_trace.json"):
    device = torch.device(f"cuda:{local_rank}")

    # 构造测试数据
    M, K, N = 16384, 16384, 16384
    A = torch.randn(M, K, dtype=torch.float16, device=device)
    B = torch.randn(K, N, dtype=torch.float16, device=device)
    comm_tensor = torch.randn(1024 * 1024 * 64, dtype=torch.float32, device=device)

    compute_stream = torch.cuda.Stream(device=device)
    comm_stream = torch.cuda.Stream(device=device)
    test_stream = torch.cuda.Stream(device=device)  # 新增测试流

    # ==========================================
    # 1. 定义需要被录制成图的核心函数
    # ==========================================
    def step_fn_compute_first(a, b, comm_t):
        """先计算后通信 (Compute -> Comm)"""
        main_stream = torch.cuda.current_stream()

        compute_stream.wait_stream(main_stream)
        comm_stream.wait_stream(main_stream)

        with torch.cuda.stream(compute_stream):
            c = torch.matmul(a, b)

        with torch.cuda.stream(comm_stream):
            dist.all_reduce(comm_t)

        main_stream.wait_stream(compute_stream)
        main_stream.wait_stream(comm_stream)
        return c

    def step_fn_comm_first(a, b, comm_t):
        """先通信后计算 (Comm -> Compute)"""
        main_stream = torch.cuda.current_stream()

        compute_stream.wait_stream(main_stream)
        comm_stream.wait_stream(main_stream)

        with torch.cuda.stream(comm_stream):
            dist.all_reduce(comm_t)

        with torch.cuda.stream(compute_stream):
            c = torch.matmul(a, b)

        main_stream.wait_stream(compute_stream)
        main_stream.wait_stream(comm_stream)
        return c

    def step_fn_test_stream(a, b, comm_t):
        """测试：在第三个独立 stream 里执行"""
        main_stream = torch.cuda.current_stream()
        test_stream.wait_stream(main_stream)

        with torch.cuda.stream(test_stream):
            c = torch.matmul(a, b)
            dist.all_reduce(comm_t)

        main_stream.wait_stream(test_stream)
        return c

    # 手动预热，确保 NCCL 环和 cuBLAS 句柄已经初始化完毕
    for _ in range(3):
        step_fn_compute_first(A, B, comm_tensor)
        step_fn_comm_first(A, B, comm_tensor)
    torch.cuda.synchronize()
    dist.barrier()

    if local_rank == 0:
        print(f"[Info] 正在录制 CUDA Graph (先计算后通信 无预留)...")

    # ==========================================
    # 2. 静态录制：先计算后通信 (无 Carveout)
    # ==========================================
    graphed_compute_first_no_carve = torch.cuda.make_graphed_callables(step_fn_compute_first, (A, B, comm_tensor))

    if local_rank == 0:
        print(f"[Info] 正在录制 CUDA Graph (先通信后计算 无预留)...")

    # ==========================================
    # 3. 静态录制：先通信后计算 (无 Carveout)
    # ==========================================
    graphed_comm_first_no_carve = torch.cuda.make_graphed_callables(step_fn_comm_first, (A, B, comm_tensor))

    if local_rank == 0:
        print(f"[Info] 正在录制 CUDA Graph (先计算后通信 预留 {sm_reserve} SM)...")

    # ==========================================
    # 4. 静态录制：先计算后通信 (有 Carveout)
    # ==========================================
    torch._C._set_sm_carveout_experimental(sm_reserve)
    graphed_compute_first_carve = torch.cuda.make_graphed_callables(step_fn_compute_first, (A, B, comm_tensor))

    if local_rank == 0:
        print(f"[Info] 正在录制 CUDA Graph (先通信后计算 预留 {sm_reserve} SM)...")

    # ==========================================
    # 5. 静态录制：先通信后计算 (有 Carveout)
    # ==========================================
    graphed_comm_first_carve = torch.cuda.make_graphed_callables(step_fn_comm_first, (A, B, comm_tensor))
    torch._C._set_sm_carveout_experimental(None)

    # ==========================================
    # 4. 启动 Profiler (可选)
    # ==========================================
    prof = None
    if use_profiler and local_rank == 0:
        os.makedirs(profile_dir, exist_ok=True)
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        )
        prof.start()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = {}

    # ------------------ Run 1: 先计算后通信 (无 carveout) ------------------
    if local_rank == 0:
        print("\n[Profile] Run 1: 先计算后通信 (无 carveout)")

    torch.cuda.synchronize()
    start_event.record()

    if prof:
        prof.step()

    with record_function("=== COMPUTE_FIRST_NO_CARVE ==="):
        graphed_compute_first_no_carve(A, B, comm_tensor)

    end_event.record()
    torch.cuda.synchronize()
    times["compute_first_no_carve"] = start_event.elapsed_time(end_event)

    # ------------------ Run 2: 先通信后计算 (无 carveout) ------------------
    if local_rank == 0:
        print("[Profile] Run 2: 先通信后计算 (无 carveout)")

    torch.cuda.synchronize()
    start_event.record()

    if prof:
        prof.step()

    with record_function("=== COMM_FIRST_NO_CARVE ==="):
        graphed_comm_first_no_carve(A, B, comm_tensor)

    end_event.record()
    torch.cuda.synchronize()
    times["comm_first_no_carve"] = start_event.elapsed_time(end_event)

    # ------------------ Run 3: 先计算后通信 (有 carveout) ------------------
    if local_rank == 0:
        print(f"[Profile] Run 3: 先计算后通信 (carveout={sm_reserve})")

    torch.cuda.synchronize()
    start_event.record()

    if prof:
        prof.step()

    with record_function("=== COMPUTE_FIRST_CARVE ==="):
        graphed_compute_first_carve(A, B, comm_tensor)

    end_event.record()
    torch.cuda.synchronize()
    times["compute_first_carve"] = start_event.elapsed_time(end_event)

    # ------------------ Run 4: 先通信后计算 (有 carveout) ------------------
    if local_rank == 0:
        print(f"[Profile] Run 4: 先通信后计算 (carveout={sm_reserve})")

    torch.cuda.synchronize()
    start_event.record()

    if prof:
        prof.step()

    with record_function("=== COMM_FIRST_CARVE ==="):
        graphed_comm_first_carve(A, B, comm_tensor)

    end_event.record()
    torch.cuda.synchronize()
    times["comm_first_carve"] = start_event.elapsed_time(end_event)

    # ==========================================
    # 6. 导出结果
    # ==========================================
    if prof:
        prof.stop()
        prof.export_chrome_trace(output_json)
        print(f"\n[Rank {local_rank}] Profile 已保存到：{output_json}")
        print(f"[Rank {local_rank}] 请在 chrome://tracing 中打开查看对比")
    elif local_rank == 0:
        print(f"\n[Rank {local_rank}] 未启用 Profiler")

    # 清理 Graph 引用释放显存
    del graphed_compute_first_no_carve
    del graphed_comm_first_no_carve
    del graphed_compute_first_carve
    del graphed_comm_first_carve
    torch.cuda.empty_cache()

    return times

def main():
    local_rank = setup()

    if local_rank == 0:
        print("="*60)
        print("🚀 终极测试：CUDA Graph + SM Carveout 完美重叠")
        print("="*60)
        print(f"   SM Carveout 值：{args.carveout}")
        if args.profile:
            print(f"   Profile 输出：{args.profile_json}")
        print("="*60)

    times = run_profile_graphs(
        local_rank,
        sm_reserve=args.carveout,
        use_profiler=args.profile,
        profile_dir=args.profile_dir,
        output_json=args.profile_json
    )

    if local_rank == 0 and times:
        print(f"\n📊 [基准测试结果 (Graph Replay Time)]")
        print(f"\n--- 无 Carveout ---")
        print(f"  先计算后通信：{times['compute_first_no_carve']:.2f} ms")
        print(f"  先通信后计算：{times['comm_first_no_carve']:.2f} ms")
        print(f"\n--- 有 Carveout (预留 {args.carveout} SM) ---")
        print(f"  先计算后通信：{times['compute_first_carve']:.2f} ms")
        print(f"  先通信后计算：{times['comm_first_carve']:.2f} ms")
        print("="*60)

    # 确保所有 GPU 任务完成
    torch.cuda.synchronize()
    dist.barrier()
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUDA Graph + SM Carveout 测试")
    parser.add_argument("--carveout", type=int, default=4,
                        help="预留的 SM 数量 (默认：4)")
    parser.add_argument("--profile", action="store_true",
                        help="启用 PyTorch Profiler 导出 trace")
    parser.add_argument("--profile-dir", type=str, default="./profile_logs",
                        help="Profile 临时目录 (默认：./profile_logs)")
    parser.add_argument("--profile-json", type=str, default="graph_trace.json",
                        help="导出的 JSON 文件路径 (默认：graph_trace.json)")
    args = parser.parse_args()
    main()
