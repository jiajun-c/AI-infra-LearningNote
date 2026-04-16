# Torch greenctx机制

greenctx是cuda13.0+中引入的新特性，其可以对sm资源进行一些隔离和管理

```shell
Python API 层：torch/cuda/green_contexts.py
     ↓
PyBind11 绑定层：torch/csrc/cuda/GreenContext.cpp
     ↓
C++ API 层：aten/src/ATen/cuda/CUDAGreenContext.h
     ↓
C++ 实现层：aten/src/ATen/cuda/CUDAGreenContext.cpp
```

## 1. Python API层
```cpp
import torch


__all__ = [
    "GreenContext",
]

_GreenContext = object
SUPPORTED = False

if hasattr(torch._C, "_CUDAGreenContext"):
    _GreenContext = torch._C._CUDAGreenContext  # type: ignore[misc]
    SUPPORTED = True


# Python shim helps Sphinx process docstrings more reliably.
# pyrefly: ignore [invalid-inheritance]
class GreenContext(_GreenContext):
    r"""Wrapper around a CUDA green context.

    .. warning::
       This API is in beta and may change in future releases.
    """

    @staticmethod
    def create(
        *,
        num_sms: int | None = None,
        workqueue_scope: str | None = None,
        workqueue_concurrency_limit: int | None = None,
        device_id: int | None = None,
    ) -> _GreenContext:
        r"""Create a CUDA green context.

        At least one of ``num_sms`` or ``workqueue_scope`` must be specified.
        Both can be combined to partition SMs and configure workqueues in the
        same green context.

        Arguments:
            num_sms (int, optional): The number of SMs to use in the green
                context. When ``None``, SMs are not partitioned.
            workqueue_scope (str, optional): Workqueue sharing scope. One of
                ``"device_ctx"`` (shared across all contexts, default driver
                behaviour) or ``"balanced"`` (non-overlapping workqueues with
                other balanced green contexts). When ``None``, no workqueue
                configuration is applied.
            workqueue_concurrency_limit (int, optional): Maximum number of
                concurrent stream-ordered workloads for the workqueue. Requires
                ``workqueue_scope`` to be set.
            device_id (int, optional): The device index of green context.
                When ``None``, the current device is used.
        """
        if not SUPPORTED:
            raise RuntimeError("PyTorch was not built with Green Context support!")
        return _GreenContext.create(  # type: ignore[attr-defined]
            device_id=device_id,
            num_sms=num_sms,
            workqueue_scope=workqueue_scope,
            workqueue_concurrency_limit=workqueue_concurrency_limit,
        )

    @staticmethod
    def max_workqueue_concurrency(device_id: int | None = None) -> int:
        r"""Return the maximum workqueue concurrency limit for the device.

        This queries the device for the default number of concurrent
        stream-ordered workloads supported by workqueue configuration
        resources.

        Arguments:
            device_id (int, optional): The device index to query. When
                ``None``, the current device is used.
        """
        if not SUPPORTED:
            raise RuntimeError("PyTorch was not built with Green Context support!")
        return _GreenContext.max_workqueue_concurrency(device_id=device_id)  # type: ignore[attr-defined]

    # Note that these functions are bypassed but we define them here
    # for Sphinx documentation purposes
    def set_context(self) -> None:  # pylint: disable=useless-parent-delegation
        r"""Make the green context the current context."""
        return super().set_context()  # type: ignore[misc]

    def pop_context(self) -> None:  # pylint: disable=useless-parent-delegation
        r"""Assuming the green context is the current context, pop it from the
        context stack and restore the previous context.
        """
        return super().pop_context()  # type: ignore[misc]

    def Stream(self) -> "torch.cuda.Stream":
        r"""Return the CUDA Stream used by the green context."""
        return super().Stream()
```

Workqueue（工作队列） 是 CUDA Driver 中用于管理 GPU 任务提交的内部队列。当 CPU 提交 CUDA kernel 到 GPU 执行时，任务会先进入 workqueue，然后由 GPU 调度执行。

创建workQueue也有两种scope的选择
- device_ctx: 默认行为，workqueue在所有的context之间共享
- balanced: 隔离模式，每个 Green Context 有非重叠的（non-overlapping） workqueue，适合进行资源隔离的场景

## 2. Pybind层
Pybind 层
```cpp
#include <ATen/cuda/CUDAGreenContext.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

// Cargo culted partially from csrc/cuda/Stream.cpp

void THCPGreenContext_init(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  py::enum_<at::cuda::WorkqueueScope>(m, "_WorkqueueScope")
      .value("device_ctx", at::cuda::WorkqueueScope::DeviceCtx)
      .value("balanced", at::cuda::WorkqueueScope::Balanced);

  py::class_<at::cuda::GreenContext>(m, "_CUDAGreenContext")
      .def_static(
          "create",
          [](std::optional<uint32_t> device_id,
             std::optional<uint32_t> num_sms,
             std::optional<std::string> workqueue_scope,
             std::optional<uint32_t> workqueue_concurrency_limit) {
            std::optional<int32_t> scope;
            if (workqueue_scope.has_value()) {
              const auto& s = *workqueue_scope;
              if (s == "device_ctx") {
                scope =
                    static_cast<int32_t>(at::cuda::WorkqueueScope::DeviceCtx);
              } else if (s == "balanced") {
                scope =
                    static_cast<int32_t>(at::cuda::WorkqueueScope::Balanced);
              } else {
                throw std::invalid_argument(
                    "workqueue_scope must be 'device_ctx' or 'balanced', got '" +
                    s + "'");
              }
            }
            return at::cuda::GreenContext::create(
                device_id, num_sms, scope, workqueue_concurrency_limit);
          },
          py::kw_only(),
          py::arg("device_id") = py::none(),
          py::arg("num_sms") = py::none(),
          py::arg("workqueue_scope") = py::none(),
          py::arg("workqueue_concurrency_limit") = py::none())
      .def_static(
          "max_workqueue_concurrency",
          &at::cuda::GreenContext::max_workqueue_concurrency,
          py::arg("device_id") = py::none())
      .def("set_context", &::at::cuda::GreenContext::setContext)
      .def("pop_context", &::at::cuda::GreenContext::popContext)
      .def("Stream", [](at::cuda::GreenContext& self) {
        auto s = self.Stream();
        cudaStream_t raw = s.stream();
        auto ptr_val = reinterpret_cast<uintptr_t>(raw);

        py::object torch_cuda = py::module::import("torch.cuda");
        py::object ExternalStream = torch_cuda.attr("ExternalStream");

        return ExternalStream(ptr_val, py::int_(s.device_index()));
      });
}

```

## 3.  C++ API 层 

这是CPP的定义
```cpp
#pragma once
#include <ATen/cuda/CUDAEvent.h>
#include <cuda.h>

// Forward declare green context as opaque ptr
typedef struct CUgreenCtx_st* CUgreenCtx;

namespace at::cuda {

namespace {
  constexpr int kStreamPerGreenContextPool = 32;
}

// Workqueue sharing scope for green contexts.
// Values match the CUDA driver API's CUdevWorkqueueConfigScope enum.
enum class WorkqueueScope : int32_t {
  DeviceCtx = 0,
  Balanced = 1,
};

class TORCH_CUDA_CPP_API GreenContext {
 public:
  static std::unique_ptr<GreenContext> create(
    std::optional<uint32_t> device_id,
    std::optional<uint32_t> num_sms,
    std::optional<int32_t> workqueue_scope = std::nullopt,
    std::optional<uint32_t> workqueue_concurrency_limit = std::nullopt);

  static uint32_t max_workqueue_concurrency(
      std::optional<uint32_t> device_id = std::nullopt);

  ~GreenContext() noexcept;

  // Delete copy constructor and assignment
  GreenContext(const GreenContext&) = delete;
  GreenContext& operator=(const GreenContext&) = delete;

  // Make this context current
  void setContext();

  void popContext();

  CUDAStream Stream();

 private:
  GreenContext(
    uint32_t device_id,
    std::optional<uint32_t> num_sms,
    std::optional<int32_t> workqueue_scope,
    std::optional<uint32_t> workqueue_concurrency_limit);

  // Implement move operations
  GreenContext(GreenContext&& other) noexcept;
  GreenContext& operator=(GreenContext&& other) noexcept;

  int32_t device_id_ = -1;
  CUgreenCtx green_ctx_ = nullptr;
  CUcontext context_ = nullptr;
  cudaStream_t parent_stream_ = nullptr;
  std::array<CUstream, kStreamPerGreenContextPool> green_ctx_streams_;
  std::atomic<int32_t> curr_stream_idx_ = -1;
};
} // namespace at::cuda

```