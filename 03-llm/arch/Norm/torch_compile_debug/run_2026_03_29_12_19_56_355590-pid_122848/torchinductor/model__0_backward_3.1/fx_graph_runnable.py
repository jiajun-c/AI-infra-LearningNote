
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config






isolate_fails_code_str = None



# torch version: 2.4.0+cu121
# torch cuda version: 12.1
# torch git version: e4ee3be4063b7c430974252fdf7db42273388d86


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Wed_Nov_22_10:17:15_PST_2023 
# Cuda compilation tools, release 12.3, V12.3.107 
# Build cuda_12.3.r12.3/compiler.33567101_0 

# GPU Hardware Info: 
# NVIDIA H100 80GB HBM3 : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, rsqrt, tangents_1):
        mul = torch.ops.aten.mul.Tensor(primals_2, rsqrt)
        mul_2 = torch.ops.aten.mul.Tensor(tangents_1, mul);  mul = None
        mul_3 = torch.ops.aten.mul.Tensor(tangents_1, primals_1);  tangents_1 = primals_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(mul_2, [0], True);  mul_2 = None
        view = torch.ops.aten.view.default(sum_1, [2048]);  sum_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, primals_2)
        mul_5 = torch.ops.aten.mul.Tensor(mul_3, rsqrt);  mul_3 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_4, [1], True);  mul_4 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_6 = torch.ops.aten.mul.Scalar(sum_2, -0.5);  sum_2 = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_6, pow_2);  mul_6 = pow_2 = None
        expand = torch.ops.aten.expand.default(mul_7, [32768, 2048]);  mul_7 = None
        div = torch.ops.aten.div.Scalar(expand, 2048);  expand = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(primals_2, 1.0);  primals_2 = None
        mul_8 = torch.ops.aten.mul.Scalar(pow_3, 2.0);  pow_3 = None
        mul_9 = torch.ops.aten.mul.Tensor(div, mul_8);  div = mul_8 = None
        add_1 = torch.ops.aten.add.Tensor(mul_5, mul_9);  mul_5 = mul_9 = None
        return [view, add_1]
        
def load_args(reader):
    buf0 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (2048,), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (32768, 2048), dtype=torch.bfloat16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 65536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (32768, 1), dtype=torch.bfloat16, is_leaf=True)  # rsqrt
    buf3 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (32768, 2048), dtype=torch.bfloat16, is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)