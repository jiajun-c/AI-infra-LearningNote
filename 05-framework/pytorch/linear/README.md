# torch linear 层

`nn.Linear`的调用链路如下所示

```shell
nn.Linear.forward(input)
  → F.linear(input, weight, bias)
    → torch._C._nn.linear(input, weight, bias)           [Python → C++ 绑定]
      → at::native::linear(input, weight, bias)          [aten/src/ATen/native/Linear.cpp:85]
        → at::addmm(bias, input, weight.t())             [主要路径，当 input.dim() == 2 且 bias 存在]
          → cuBLASLt matmul                              [aten/src/ATen/cuda/CUDABlas.cpp]
            → 检查 at::globalContext()._SMCarveout_EXPERIMENTAL()
```

## 1. Python API: 

torch/nn/modules/linear.py:134
```python3
    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.linear(input, self.weight, self.bias)
```

## 2. 函数绑定

torch/nn/functional.py:2328-2329

```python3
linear = _add_docstr(
    torch._C._nn.linear,
    r"""
```

## 3. C++实现

`at::addmm` 或 `at::matmul`

```cpp
Tensor linear(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt) {
  // _matmul_impl checks this again later, but _flatten_nd_linear does not work on scalars inputs,
  // so let's try to catch this here already
  const auto input_dim = input.dim();
  const auto weight_dim = weight.dim();
  TORCH_CHECK(input_dim != 0 && weight_dim != 0,
              "both arguments to linear need to be at least 1D, but they are ",
              input_dim, "D and ", weight_dim, "D");

  // See [Note: hacky wrapper removal for optional tensor]
  auto bias = bias_opt.has_value()
    ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
    : c10::MaybeOwned<Tensor>::owned(std::in_place);
  if (input.is_mkldnn()) {
    return at::mkldnn_linear(input, weight, *bias);
  }
#if defined(C10_MOBILE)
  if (xnnpack::use_linear(input, weight, *bias)) {
    return xnnpack::linear(input, weight, *bias);
  }
#endif
  if (input_dim == 2 && bias->defined()) {
    // Fused op is marginally faster.
    return at::addmm(*bias, input, weight.t());
  }

  const auto is_bias_likely_fusable = (
      bias->defined() &&
      // cuBLASLt: will fuse in the epilogue without copies
      // when input/weight/bias are all strided.
      // When weight is not strided, bias will not be fused,
      // but we can still dispatch here to avoid at::matmul
      // path which will probably use a very similar
      // flattening optimization.
      ((bias->dim() == 1 || bias->squeeze().dim() == 1) && bias->is_contiguous_or_false())
  );
  if (is_bias_likely_fusable && !input.is_xla()) {
    // Also hit the fused path for contiguous nD input, if not using xla
    // backend. Reshaping/flattening has some performance implications on xla.
    if (input.is_contiguous_or_false()) {
      return _flatten_nd_linear(input, weight, *bias);
    } else if (parseLinearFlatten3d()) {
      // If user forces flattening via env var
      const Tensor input_cont = input.contiguous();
      return _flatten_nd_linear(input_cont, weight, *bias);
    }
  }
  auto output = at::matmul(input, weight.t());
  if (bias->defined()) {
    // for composite compliance use out-of-place version of `add`
    if (isTensorSubclassLike(*bias) ||
        bias->_fw_grad(/*level*/ 0).defined()) {
      output = at::add(output, *bias);
    } else {
      output.add_(*bias);
    }
  }
  return output;
}
```

## 4. CUDA BLAS

aten/src/ATen/cuda/CUDABlas.cpp

```cpp
#ifndef USE_ROCM
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    computeDesc.setAttribute<int32_t>(
        CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount -
            at::globalContext()._SMCarveout_EXPERIMENTAL().value());
  }
#else
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    stream = _getCarveoutStream(
        at::globalContext()._SMCarveout_EXPERIMENTAL().value());
    _syncCurrentWithCarveoutStream(stream, true);
  }
#endif
```
