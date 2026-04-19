import torch
import torch.nn as nn
import gc

def snapshot_memory(prefix=""):
    """打印当前显存使用情况"""
    gc.collect()
    torch.cuda.empty_cache()

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    print(f"[{prefix}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    return allocated, reserved


class SimpleModel(nn.Module):
    """一个简单的 MLP 模型用于演示"""
    def __init__(self, input_dim=1000, hidden_dim=4096, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print("=" * 60)
    print("PyTorch 显存监控 Demo")
    print("=" * 60)

    # 1. 初始状态
    snapshot_memory("初始状态")

    # 2. 加载模型
    print("\n>>> 加载模型...")
    model = SimpleModel().cuda()
    snapshot_memory("加载模型后")

    # 3. 创建输入张量
    print("\n>>> 创建输入张量 (batch=10000)...")
    positions = torch.randn(10000, 1000, requires_grad=True).cuda()
    snapshot_memory("创建输入后")

    # 4. 前向传播
    print("\n>>> 执行前向传播...")
    energy = model(positions)
    snapshot_memory("前向传播后")  # 如果显存激增，说明缓存了多余激活

    # 5. 反向传播
    print("\n>>> 执行反向传播...")
    force = -torch.autograd.grad(energy.sum(), positions, retain_graph=False)[0]
    snapshot_memory("反向传播后")

    # 6. 清理梯度
    print("\n>>> 清理梯度...")
    model.zero_grad(set_to_none=True)
    snapshot_memory("清理梯度后")

    # 7. 删除大张量
    print("\n>>> 删除输入张量...")
    del positions, energy, force
    snapshot_memory("删除张量后")

    # 8. 清空缓存
    print("\n>>> 清空 CUDA 缓存...")
    torch.cuda.empty_cache()
    snapshot_memory("清空缓存后")

    # 9. 演示 grad 上下文（不保存计算图）
    print("\n>>> 使用 torch.no_grad() 进行推理...")
    with torch.no_grad():
        test_input = torch.randn(10000, 1000).cuda()
        output = model(test_input)
        snapshot_memory("no_grad 推理后")
        del test_input, output

    # 10. 对比：保存计算图的情况
    print("\n>>> 对比：retain_graph=True 的情况...")
    positions2 = torch.randn(10000, 1000, requires_grad=True).cuda()
    energy2 = model(positions2)
    force2 = -torch.autograd.grad(energy2.sum(), positions2, retain_graph=True)[0]
    snapshot_memory("retain_graph=True 后")  # 显存不会完全释放

    # 手动清理
    del positions2, energy2, force2
    gc.collect()
    torch.cuda.empty_cache()
    snapshot_memory("最终清理后")

    print("\n" + "=" * 60)
    print("Demo 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
