class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[2048]", primals_2: "bf16[32768, 2048]", rsqrt: "bf16[32768, 1]", tangents_1: "bf16[32768, 2048]"):
        # File: /usr/local/lib/python3.10/dist-packages/torch/_dynamo/external_utils.py:38 in inner, code: return fn(*args, **kwargs)
        mul: "bf16[32768, 2048]" = torch.ops.aten.mul.Tensor(primals_2, rsqrt)
        mul_2: "bf16[32768, 2048]" = torch.ops.aten.mul.Tensor(tangents_1, mul);  mul = None
        mul_3: "bf16[32768, 2048]" = torch.ops.aten.mul.Tensor(tangents_1, primals_1);  tangents_1 = primals_1 = None
        sum_1: "bf16[1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_2, [0], True);  mul_2 = None
        view: "bf16[2048]" = torch.ops.aten.view.default(sum_1, [2048]);  sum_1 = None
        mul_4: "bf16[32768, 2048]" = torch.ops.aten.mul.Tensor(mul_3, primals_2)
        mul_5: "bf16[32768, 2048]" = torch.ops.aten.mul.Tensor(mul_3, rsqrt);  mul_3 = None
        sum_2: "bf16[32768, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [1], True);  mul_4 = None
        pow_2: "bf16[32768, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_6: "bf16[32768, 1]" = torch.ops.aten.mul.Scalar(sum_2, -0.5);  sum_2 = None
        mul_7: "bf16[32768, 1]" = torch.ops.aten.mul.Tensor(mul_6, pow_2);  mul_6 = pow_2 = None
        expand: "bf16[32768, 2048]" = torch.ops.aten.expand.default(mul_7, [32768, 2048]);  mul_7 = None
        div: "bf16[32768, 2048]" = torch.ops.aten.div.Scalar(expand, 2048);  expand = None
        pow_3: "bf16[32768, 2048]" = torch.ops.aten.pow.Tensor_Scalar(primals_2, 1.0);  primals_2 = None
        mul_8: "bf16[32768, 2048]" = torch.ops.aten.mul.Scalar(pow_3, 2.0);  pow_3 = None
        mul_9: "bf16[32768, 2048]" = torch.ops.aten.mul.Tensor(div, mul_8);  div = mul_8 = None
        
        # File: /usr/local/lib/python3.10/dist-packages/torch/_dynamo/external_utils.py:38 in inner, code: return fn(*args, **kwargs)
        add_1: "bf16[32768, 2048]" = torch.ops.aten.add.Tensor(mul_5, mul_9);  mul_5 = mul_9 = None
        return [view, add_1]
        