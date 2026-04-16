import ctypes
import torch
import time

# 加载 libsmctrl
libsmctrl = ctypes.CDLL("libsmctrl.so")

def set_next_sm_count(num_sms):
    """设置下一个 kernel 使用的 SM 数量"""
    mask = ~((1 << num_sms) - 1) & 0xFFFFFFFFFFFFFFFF
    libsmctrl.libsmctrl_set_next_mask(mask)

def set_global_sm_count(num_sms):
    """设置全局默认 SM 数量"""
    mask = ~((1 << num_sms) - 1) & 0xFFFFFFFFFFFFFFFF
    libsmctrl.libsmctrl_set_global_mask(mask)

def run_elementwise_test():
    """测试 element-wise kernel 在不同 SM 数量下的表现"""
    # 准备数据
    size = 100_000_000
    x = torch.randn(size).cuda()
    y = torch.randn(size).cuda()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"数据大小：{size / 1e6:.0f}M 元素")
    print()

    # 全局预热：先用全部 SM 跑几次
    print("Warmup: 全局预热 5 次...")
    set_global_sm_count(66)
    for _ in range(5):
        z = x + y
        z = z * 2.0
        z = torch.sin(z)
    torch.cuda.synchronize()

    # 测试不同 SM 数量
    for sm_count in [66, 32, 16, 8]:
        print(f"\n测试 SM 数量：{sm_count}")
        set_global_sm_count(sm_count)

        # 每个配置先 warmup 5 次
        for _ in range(5):
            z = x + y
            z = z * 2.0
            z = torch.sin(z)
        torch.cuda.synchronize()

        # 正式计时 10 次
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(10):
            z = x + y
            z = z * 2.0
            z = torch.sin(z)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"SM 数量：{sm_count:2d} | 耗时：{elapsed*1000:.2f} ms ({elapsed*100:.2f} ms/iter)")

    # 恢复全部 SM
    set_global_sm_count(66)
    print("\n已恢复使用全部 66 个 SM")


def run_single_kernel_for_profile():
    """运行单个 kernel 用于 NCU profile"""
    size = 100_000_000
    x = torch.randn(size).cuda()
    y = torch.randn(size).cuda()

    # 预热
    for _ in range(10):
        z = x + y
        z = z * 2.0
    torch.cuda.synchronize()

    # 执行目标 kernel
    result = torch.sin(x + y)
    torch.cuda.synchronize()
    print("Kernel 执行完成")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--profile":
        # Profile 模式：只运行一个 kernel，方便 NCU 捕获
        sm_count = int(sys.argv[2]) if len(sys.argv) > 2 else 32
        print(f"设置 SM 数量：{sm_count}")
        set_global_sm_count(sm_count)

        # 创建标记，方便在 NCU 中识别
        print(f"[NCU_MARKER] SM_COUNT={sm_count}")

        run_single_kernel_for_profile()

        print("[NCU_MARKER] DONE")
    else:
        # 正常测试模式
        run_elementwise_test()

if __name__ == "__main__":
    run_elementwise_test()
