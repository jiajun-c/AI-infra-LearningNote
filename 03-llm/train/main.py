"""
基础模型实现 - 不使用数据并行
使用 torch.autograd.Function 手写前向传播和反向传播

包含：
1. 手写算子的实现（Linear, ReLU, CrossEntropyLoss）
2. 正确性验证（与 PyTorch 原生实现对比）
3. 性能测试
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import sys
import os

# 添加路径以导入 DP 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============== 手写算子实现 ==============

class LinearFunction(torch.autograd.Function):
    """
    手写线性层的前向和反向传播
    y = x @ W.t() + b
    """
    @staticmethod
    def forward(ctx, x, weight, bias):
        """
        前向传播
        ctx: 上下文对象，用于保存反向传播需要的信息
        """
        # 使用 save_for_backward 保存反向传播需要的张量
        ctx.save_for_backward(x, weight, bias)
        output = x @ weight.t() + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播
        grad_output: 上一层传来的梯度
        """
        # 从 saved_tensors 中取出前向传播保存的中间结果
        x, weight, bias = ctx.saved_tensors

        # 计算各参数的梯度
        # y = x @ W.t() + b
        # dy/dx = grad_output @ W
        # dy/dW = grad_output.t() @ x
        # dy/db = grad_output.sum(dim=0)
        grad_x = grad_output @ weight
        grad_weight = grad_output.t() @ x
        grad_bias = grad_output.sum(dim=0)

        return grad_x, grad_weight, grad_bias


class ReLUFunction(torch.autograd.Function):
    """
    手写 ReLU 的前向和反向传播
    """
    @staticmethod
    def forward(ctx, x):
        # 使用 save_for_backward 保存输入用于反向传播
        ctx.save_for_backward(x)
        return torch.maximum(x, torch.zeros_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # ReLU 的梯度: x > 0 时为 1，否则为 0
        grad_input = grad_output.clone()
        grad_input[x <= 0] = 0
        return grad_input


class CrossEntropyLossFunction(torch.autograd.Function):
    """
    手写 CrossEntropyLoss 的前向和反向传播
    结合了 LogSoftmax 和 NLLLoss
    """
    @staticmethod
    def forward(ctx, logits, targets):
        """
        前向传播
        logits: [batch_size, num_classes]
        targets: [batch_size] 整数标签
        """
        batch_size = logits.shape[0]

        # 数值稳定的 log softmax
        x_max = logits.max(dim=-1, keepdim=True)[0]
        exp_logits = torch.exp(logits - x_max)
        sum_exp = exp_logits.sum(dim=-1, keepdim=True)
        log_probs = logits - x_max - torch.log(sum_exp)

        # 计算损失: -log_probs[range(batch), targets] 的平均值
        loss = -log_probs[torch.arange(batch_size, device=logits.device), targets].mean()

        # 保存反向传播需要的信息
        ctx.save_for_backward(log_probs, targets)
        ctx.batch_size = batch_size

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播
        grad_output: 标量损失的梯度，通常为 1
        """
        log_probs, targets = ctx.saved_tensors
        batch_size = ctx.batch_size

        # Softmax + CrossEntropy 的梯度
        # grad = softmax - one_hot(targets) / batch_size
        probs = torch.exp(log_probs)
        grad_logits = probs.clone()
        grad_logits[torch.arange(batch_size, device=probs.device), targets] -= 1
        grad_logits = grad_logits / batch_size

        # 乘以传入的梯度
        grad_logits = grad_logits * grad_output

        return grad_logits, None  # targets 不需要梯度


class HandwrittenLinear(nn.Module):
    """使用手写 Function 的线性层"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (2.0 / in_features) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return LinearFunction.apply(x, self.weight, self.bias)


class HandwrittenReLU(nn.Module):
    """使用手写 Function 的 ReLU"""
    def forward(self, x):
        return ReLUFunction.apply(x)


class HandwrittenCrossEntropyLoss(nn.Module):
    """使用手写 Function 的 CrossEntropyLoss"""
    def forward(self, logits, targets):
        return CrossEntropyLossFunction.apply(logits, targets)


class SimpleMLP(nn.Module):
    """
    使用手写算子的 MLP 模型
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = HandwrittenLinear(input_dim, hidden_dim)
        self.relu1 = HandwrittenReLU()
        self.linear2 = HandwrittenLinear(hidden_dim, hidden_dim)
        self.relu2 = HandwrittenReLU()
        self.linear3 = HandwrittenLinear(hidden_dim, output_dim)
        self.loss_fn = HandwrittenCrossEntropyLoss()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

    def compute_loss(self, logits, targets):
        return self.loss_fn(logits, targets)


class PyTorchMLP(nn.Module):
    """
    PyTorch 原生实现的 MLP 模型（用于正确性验证）
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

    def compute_loss(self, logits, targets):
        return self.loss_fn(logits, targets)


# ============== 验证函数 ==============

def verify_correctness():
    """
    验证手写算子的正确性
    对比手写实现与 PyTorch 原生实现的输出和梯度
    """
    print("=" * 60)
    print("Correctness Verification: Handwritten vs PyTorch Native")
    print("=" * 60)

    device = 'cpu'  # 在 CPU 上验证，确保一致性
    torch.manual_seed(42)  # 设置随机种子确保可重复性

    input_dim = 784
    hidden_dim = 256
    output_dim = 10
    batch_size = 32

    # 创建输入数据
    x = torch.randn(batch_size, input_dim, device=device)
    y = torch.randint(0, output_dim, (batch_size,), device=device)

    # 创建手写模型
    model_handwritten = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)

    # 创建 PyTorch 原生模型，并复制权重
    model_pytorch = PyTorchMLP(input_dim, hidden_dim, output_dim).to(device)

    # 复制权重（确保两个模型使用相同的参数）
    with torch.no_grad():
        model_pytorch.linear1.weight.copy_(model_handwritten.linear1.weight)
        model_pytorch.linear1.bias.copy_(model_handwritten.linear1.bias)
        model_pytorch.linear2.weight.copy_(model_handwritten.linear2.weight)
        model_pytorch.linear2.bias.copy_(model_handwritten.linear2.bias)
        model_pytorch.linear3.weight.copy_(model_handwritten.linear3.weight)
        model_pytorch.linear3.bias.copy_(model_handwritten.linear3.bias)

    # 1. 验证前向传播
    print("\n[1] Forward Pass Verification")
    model_handwritten.eval()
    model_pytorch.eval()

    with torch.no_grad():
        out_handwritten = model_handwritten(x)
        out_pytorch = model_pytorch(x)

    forward_diff = (out_handwritten - out_pytorch).abs().max().item()
    print(f"   Max forward output difference: {forward_diff:.2e}")
    print(f"   Forward pass: {'✓ PASS' if forward_diff < 1e-5 else '✗ FAIL'}")

    # 2. 验证损失计算
    print("\n[2] Loss Calculation Verification")
    loss_handwritten = model_handwritten.compute_loss(out_handwritten, y)
    loss_pytorch = model_pytorch.compute_loss(out_pytorch, y)

    loss_diff = abs(loss_handwritten.item() - loss_pytorch.item())
    print(f"   Handwritten loss: {loss_handwritten.item():.6f}")
    print(f"   PyTorch loss:     {loss_pytorch.item():.6f}")
    print(f"   Loss difference:  {loss_diff:.2e}")
    print(f"   Loss calculation: {'✓ PASS' if loss_diff < 1e-5 else '✗ FAIL'}")

    # 3. 验证反向传播（梯度）
    print("\n[3] Backward Pass Verification")

    # 重新前向传播（需要梯度）
    model_handwritten.train()
    model_pytorch.train()

    # 清零梯度
    model_handwritten.zero_grad()
    model_pytorch.zero_grad()

    # 前向传播
    out_handwritten = model_handwritten(x)
    out_pytorch = model_pytorch(x)

    # 计算损失
    loss_handwritten = model_handwritten.compute_loss(out_handwritten, y)
    loss_pytorch = model_pytorch.compute_loss(out_pytorch, y)

    # 反向传播
    loss_handwritten.backward()
    loss_pytorch.backward()

    # 比较梯度
    print("   Gradient comparison for each layer:")

    grad_diffs = []
    for i, (hw_layer, pt_layer) in enumerate([
        (model_handwritten.linear1, model_pytorch.linear1),
        (model_handwritten.linear2, model_pytorch.linear2),
        (model_handwritten.linear3, model_pytorch.linear3),
    ]):
        weight_grad_diff = (hw_layer.weight.grad - pt_layer.weight.grad).abs().max().item()
        bias_grad_diff = (hw_layer.bias.grad - pt_layer.bias.grad).abs().max().item()
        grad_diffs.extend([weight_grad_diff, bias_grad_diff])
        print(f"   Layer {i+1} - Weight grad diff: {weight_grad_diff:.2e}, Bias grad diff: {bias_grad_diff:.2e}")

    max_grad_diff = max(grad_diffs)
    print(f"\n   Max gradient difference: {max_grad_diff:.2e}")
    print(f"   Backward pass: {'✓ PASS' if max_grad_diff < 1e-5 else '✗ FAIL'}")

    # 4. 验证训练过程（多个 step）
    print("\n[4] Training Process Verification (5 steps)")

    optimizer_hw = optim.SGD(model_handwritten.parameters(), lr=0.01)
    optimizer_pt = optim.SGD(model_pytorch.parameters(), lr=0.01)

    losses_hw = []
    losses_pt = []

    for step in range(5):
        # 清零梯度
        optimizer_hw.zero_grad()
        optimizer_pt.zero_grad()

        # 前向传播
        out_hw = model_handwritten(x)
        out_pt = model_pytorch(x)

        # 计算损失
        loss_hw = model_handwritten.compute_loss(out_hw, y)
        loss_pt = model_pytorch.compute_loss(out_pt, y)

        losses_hw.append(loss_hw.item())
        losses_pt.append(loss_pt.item())

        # 反向传播
        loss_hw.backward()
        loss_pt.backward()

        # 更新参数
        optimizer_hw.step()
        optimizer_pt.step()

    loss_track_diff = max(abs(hw - pt) for hw, pt in zip(losses_hw, losses_pt))
    print(f"   Max loss tracking difference: {loss_track_diff:.2e}")
    print(f"   Training process: {'✓ PASS' if loss_track_diff < 1e-5 else '✗ FAIL'}")

    print("\n" + "=" * 60)
    print("Correctness verification completed!")
    print("=" * 60)

    return forward_diff < 1e-5 and max_grad_diff < 1e-5


def benchmark_performance():
    """
    性能测试：对比手写实现与 PyTorch 原生实现的性能
    """
    print("\n" + "=" * 60)
    print("Performance Benchmark: Handwritten vs PyTorch Native")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    input_dim = 784
    hidden_dim = 512
    output_dim = 10
    batch_size = 128
    num_iterations = 100

    # 创建输入数据
    x = torch.randn(batch_size, input_dim, device=device)
    y = torch.randint(0, output_dim, (batch_size,), device=device)

    # 测试手写实现
    print("\n[1] Handwritten Implementation")
    model_hw = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)
    optimizer_hw = optim.SGD(model_hw.parameters(), lr=0.01)

    # 预热
    for _ in range(10):
        optimizer_hw.zero_grad()
        out = model_hw(x)
        loss = model_hw.compute_loss(out, y)
        loss.backward()
        optimizer_hw.step()

    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_iterations):
        optimizer_hw.zero_grad()
        out = model_hw(x)
        loss = model_hw.compute_loss(out, y)
        loss.backward()
        optimizer_hw.step()

    if device == 'cuda':
        torch.cuda.synchronize()
    hw_time = time.time() - start_time

    print(f"   Total time: {hw_time:.4f}s")
    print(f"   Time per iteration: {hw_time / num_iterations * 1000:.2f}ms")

    # 测试 PyTorch 原生实现
    print("\n[2] PyTorch Native Implementation")
    model_pt = PyTorchMLP(input_dim, hidden_dim, output_dim).to(device)
    optimizer_pt = optim.SGD(model_pt.parameters(), lr=0.01)

    # 预热
    for _ in range(10):
        optimizer_pt.zero_grad()
        out = model_pt(x)
        loss = model_pt.compute_loss(out, y)
        loss.backward()
        optimizer_pt.step()

    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_iterations):
        optimizer_pt.zero_grad()
        out = model_pt(x)
        loss = model_pt.compute_loss(out, y)
        loss.backward()
        optimizer_pt.step()

    if device == 'cuda':
        torch.cuda.synchronize()
    pt_time = time.time() - start_time

    print(f"   Total time: {pt_time:.4f}s")
    print(f"   Time per iteration: {pt_time / num_iterations * 1000:.2f}ms")

    # 对比结果
    print("\n[3] Comparison")
    print(f"   Speed ratio (Handwritten/PyTorch): {hw_time / pt_time:.2f}x")
    if hw_time > pt_time:
        print(f"   PyTorch native is {hw_time / pt_time:.2f}x faster")
    else:
        print(f"   Handwritten is {pt_time / hw_time:.2f}x faster")

    print("\n" + "=" * 60)
    print("Performance benchmark completed!")
    print("=" * 60)


def train():
    """训练函数"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 超参数
    input_dim = 784
    hidden_dim = 256
    output_dim = 10
    learning_rate = 0.01
    batch_size = 64
    num_epochs = 10

    # 创建模型
    model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)

    # 使用标准 SGD 优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # 创建数据集
    X = torch.randn(10000, input_dim)
    y = torch.randint(0, output_dim, (10000,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            logits = model(batch_x)

            # 计算损失
            loss = model.compute_loss(logits, batch_y)
            total_loss += loss.item()

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 计算准确率
            predictions = logits.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    print("Training completed!")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='Handwritten Operators Demo')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'verify', 'benchmark', 'all'],
                        help='Running mode')
    args = parser.parse_args()

    if args.mode == 'train':
        print("=" * 60)
        print("Training with Handwritten Operators")
        print("=" * 60)
        train()
    elif args.mode == 'verify':
        verify_correctness()
    elif args.mode == 'benchmark':
        benchmark_performance()
    elif args.mode == 'all':
        verify_correctness()
        benchmark_performance()
        print("\n")
        train()


if __name__ == "__main__":
    main()