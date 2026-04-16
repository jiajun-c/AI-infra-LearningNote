"""
CUDA Green Context 示例代码

Green Context 是 CUDA 13.x 引入的新特性，用于：
1. 预分配 SM 资源给特定上下文
2. 实现确定性的资源隔离
3. 在同一进程内动态划分 SM 资源

注意：PyTorch 2.11+ 才支持 torch.cuda.green_contexts API
"""

import torch

# 检查 PyTorch 版本
torch_version = tuple(map(int, torch.__version__.split('+')[0].split('.')))
print(f"PyTorch version: {torch.__version__}")
print(f"Required: 2.11+")

if torch_version < (2, 11):
    print("\n⚠️  当前 PyTorch 版本不支持 GreenContext API")
    print("\n可选方案:")
    print("1. 升级 PyTorch: pip install --upgrade torch")
    print("2. 使用 CUDA 13.x C API 直接调用 (需要 CUDA 13.1+)")
    print("3. 使用 CUDA Stream 作为替代方案")

    # 使用标准 CUDA Stream 作为替代
    print("\n--- 使用标准 CUDA Stream 示例 ---")
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    with torch.cuda.stream(stream1):
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        c = a @ b

    with torch.cuda.stream(stream2):
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = x @ y

    stream1.synchronize()
    stream2.synchronize()
    print("标准 Stream 执行完成")

else:
    # PyTorch 2.11+ 使用 GreenContext
    print("\n✓ 当前 PyTorch 版本支持 GreenContext")

    from torch.cuda.green_contexts import GreenContext

    # 创建一个占用 32 个 SM 的 Green Context
    green_ctx = GreenContext.create(num_sms=32, device_id=0)
    print(f"GreenContext 创建成功：{green_ctx}")

    # 获取这个上下文中专属的 Stream
    stream = green_ctx.create_stream()
    print(f"Stream 创建成功：{stream}")

    # 在 Green Context Stream 上执行操作
    with torch.cuda.stream(stream):
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        c = a @ b

    stream.synchronize()
    print("Green Context Stream 执行完成")
