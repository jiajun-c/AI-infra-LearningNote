import torch

# 模拟全局状态字典
STATE = {}

class DummyAsyncComm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, layer_id):
        # 模拟申请显存
        out = torch.empty_like(x) 
        out.copy_(x)
        
        ptr = out.data_ptr()
        
        # 💣 致命操作：把物理地址存入字典
        STATE[ptr] = f"【Layer {layer_id} 的专属状态】"
        
        # 重点：我们在 Python 层面加一个 Print，看看它什么时候执行
        print(f"    -> [Python 解释器执行] Layer {layer_id} 申请地址: {hex(ptr)}")
        
        ctx.ptr = ptr
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None

def run_model(x):
    loss = 0
    for i in range(3):
        x = DummyAsyncComm.apply(x, i)
        x = x + 1.0  # 释放上一步的张量
        loss = loss + x.sum()
    return loss

# =====================================================================
# 准备环境
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print("本测试必须在 CUDA 环境下运行！")
    exit()

x = torch.randn(1024, 1024, device=device, requires_grad=True)

# =====================================================================
print("\n" + "="*50)
print(" 第一场：标准的 Eager 模式 (惨剧发生)")
print("="*50)
STATE.clear()

run_model(x)

print("\n[Eager 结果] 看看字典里有几个键？")
for k, v in STATE.items():
    print(f"  地址 {hex(k)} : {v}")
print("结论：地址复用，Layer 0 和 Layer 1 的状态被无情覆盖！")


# =====================================================================
print("\n\n" + "="*50)
print(" 第二场：CUDA Graph 模式 (奇迹与炸弹)")
print("="*50)
STATE.clear()

# CUDA Graph 需要一块静态的内存池，我们先进行预热 (Warmup)
# 注意：预热阶段依然使用的是 Eager 显存分配器，所以我们通常不管预热阶段的状态
warmup_stream = torch.cuda.Stream()
with torch.cuda.stream(warmup_stream):
    run_model(x)
torch.cuda.current_stream().wait_stream(warmup_stream)

# 清空预热时弄脏的字典
STATE.clear()

# ⚡️ 核心环节：开始录制 CUDA Graph ⚡️
print("\n--- 🎬 阶段 A：开始录制 (Capture) ---")
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_loss = run_model(x)

print("\n[Capture 结果] 看看录制时字典里存了几个键？")
for k, v in STATE.items():
    print(f"  地址 {hex(k)} : {v}")
print("结论：奇迹发生！Graph 显存池为它们分配了独立的连续地址，没有复用，字典保住了！")


# ⚡️ 致命环节：重放 CUDA Graph ⚡️
print("\n--- 🚀 阶段 B：线上真实推理/训练 (Replay) ---")
# 假设我们在下一轮 Iteration，输入了新的数据，我们要重新存入状态
STATE.clear() 

print("正在执行 g.replay()...")
g.replay() # 极速执行底层 CUDA Kernel
torch.cuda.synchronize()

print("\n[Replay 结果] 执行完了，看看字典里有几个键？")
if len(STATE) == 0:
    print("  -> 字典是空的！！！")
else:
    for k, v in STATE.items():
        print(f"  地址 {hex(k)} : {v}")

print("\n💀 终极结论：")
print("在 g.replay() 时，没有任何一条 Print 被打印，字典也没有被写入！")
print("因为 CUDA Graph 彻底绕过了 Python 解释器。你的异步通信逻辑在真实运行时成了『植物人』！")