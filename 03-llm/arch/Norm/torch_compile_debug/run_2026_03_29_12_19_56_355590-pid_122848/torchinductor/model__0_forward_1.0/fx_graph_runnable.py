
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


torch._functorch.config.debug_partitioner = True
torch._functorch.config.unlift_effect_tokens = True



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

    
    
    def forward(self, primals_1, primals_2):
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(primals_2, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [1], True);  pow_1 = None
        add = torch.ops.aten.add.Scalar(mean, 9.98377799987793e-07);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        mul = torch.ops.aten.mul.Tensor(primals_2, rsqrt)
        mul_1 = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
        return [mul_1, primals_1, primals_2, rsqrt]
        
def load_args(reader):
    buf0 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (2048,), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (32768, 2048), dtype=torch.bfloat16, is_leaf=True)  # primals_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)