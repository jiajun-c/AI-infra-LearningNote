import torch

# 模拟你的全局状态字典 _A2A_OVERLAP_STATE
STATE = {}

class DummyAsyncComm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, layer_id):
        # 1. 模拟申请显存 (类似于 torch.empty(recv_splits))
        # 故意申请一块比较大的显存 (4MB)，确保它会被 Caching Allocator 管理
        out = torch.empty_like(x) 
        out.copy_(x)
        
        # 2. 获取底层的物理显存地址
        ptr = out.data_ptr()
        print(f"[Forward] Layer {layer_id} 申请了显存，物理地址: {hex(ptr)}")
        
        # 3. 致命 BUG：把物理地址当成主键存入字典
        STATE[ptr] = f"我是 Layer {layer_id} 存下来的重要状态！"
        
        ctx.ptr = ptr
        ctx.layer_id = layer_id
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ptr = ctx.ptr
        
        # 反向传播时，尝试拿着当时的物理地址去取状态
        retrieved_state = STATE.get(ptr, "找不到状态 (被 pop 或根本不存在)")
        
        print(f"\n[Backward] Layer {ctx.layer_id} 拿着地址 {hex(ptr)} 尝试取回状态...")
        print(f"  -> 取回的内容: {retrieved_state}")
        
        # 检查是否取错了别人的状态
        expected_state = f"我是 Layer {ctx.layer_id} 存下来的重要状态！"
        if retrieved_state != expected_state:
            print(f"  ❌ 严重 BUG 爆发！Layer {ctx.layer_id} 的状态被覆盖了！")
        else:
            print(f"  ✅ 状态正确。")
            
        return grad_out, None

# ================= 运行测试 =================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}\n")

x = torch.randn(1024, 1024, device=device, requires_grad=True)

loss = 0
for i in range(3):
    # 经过我们的自定义 Function (类似 Layer N 的 Function1)
    x = DummyAsyncComm.apply(x, i)
    
    # 模拟一段不需要保存输入梯度的本地计算 (类似 local_compute)
    # x + 1.0 的反向传播是 dx = dout，它不需要保存前向的 x。
    # 所以在执行完这行代码后，上面 DummyAsyncComm 吐出来的 x 瞬间变成“垃圾”，被释放回显存池！
    x = x * x
    
    loss = loss + x.sum()

print("\n================ Forward 结束 ================")
print("此时全局 STATE 字典里到底存了几个键？")
for k, v in STATE.items():
    print(f"地址 {hex(k)} : {v}")
print("==============================================\n")

print("================ 开始 Backward ===============")
loss.backward(create_graph=True)

