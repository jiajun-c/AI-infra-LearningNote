class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[2048]", primals_2: "bf16[32768, 2048]"):
        # File: /usr/local/lib/python3.10/dist-packages/torch/_dynamo/external_utils.py:38 in inner, code: return fn(*args, **kwargs)
        pow_1: "bf16[32768, 2048]" = torch.ops.aten.pow.Tensor_Scalar(primals_2, 2)
        mean: "bf16[32768, 1]" = torch.ops.aten.mean.dim(pow_1, [1], True);  pow_1 = None
        add: "bf16[32768, 1]" = torch.ops.aten.add.Scalar(mean, 9.98377799987793e-07);  mean = None
        rsqrt: "bf16[32768, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "bf16[32768, 2048]" = torch.ops.aten.mul.Tensor(primals_2, rsqrt)
        mul_1: "bf16[32768, 2048]" = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
        return [mul_1, primals_1, primals_2, rsqrt]
        