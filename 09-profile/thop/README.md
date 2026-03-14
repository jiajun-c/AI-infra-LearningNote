# THOP: PyTorch OpCounter

使用thop可以对模型中的数据信息进行统计，其主要统计下面的两类信息

- MAC: 乘加操作数目
- parameter: 模型参数数量

```python3
import torch
from torchvision.models import resnet50  # Example model

from thop import profile  # Import the profile function from THOP

# Load a pre-trained model (e.g., ResNet50)
model = resnet50()

# Create a dummy input tensor matching the model's expected input shape
dummy_input = torch.randn(1, 3, 224, 224)

# Profile the model
macs, params = profile(model, inputs=(dummy_input,))

print(f"MACs: {macs}, Parameters: {params}")
# Expected output: MACs: 4139975680.0, Parameters: 25557032.0
```

对于不原生支持thop的模型我们可以选择自定义计数器来进行统计


```python3
import torch
import torch.nn as nn

from thop import profile


# Define your custom module
class YourCustomModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers, e.g., a convolution
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)


# Define a custom counting function for your module
# This function should calculate and return the MACs for the module's operations
def count_your_custom_module(module, x, y):
    # Example: Calculate MACs for the conv layer
    # Note: This is a simplified example. Real calculations depend on the module's specifics.
    # MACs = output_height * output_width * kernel_height * kernel_width * in_channels * out_channels
    # For simplicity, we'll just assign a placeholder value or use a helper if available
    # In a real scenario, you'd implement the precise MAC calculation here.
    # For nn.Conv2d, THOP usually handles it, but this demonstrates the concept.
    macs = 0  # Placeholder: Implement actual MAC calculation based on module logic
    # You might need access to module properties like kernel_size, stride, padding, channels etc.
    # Example for a Conv2d layer (simplified):
    if isinstance(module, nn.Conv2d):
        _, _, H, W = y.shape  # Output shape
        k_h, k_w = module.kernel_size
        in_c = module.in_channels
        out_c = module.out_channels
        groups = module.groups
        macs = (k_h * k_w * in_c * out_c * H * W) / groups
    module.total_ops += torch.DoubleTensor([macs])  # Accumulate MACs


# Instantiate a model containing your custom module
model = YourCustomModule()  # Or a larger model incorporating this module

# Create a dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Profile the model, providing the custom operation mapping
macs, params = profile(model, inputs=(dummy_input,), custom_ops={YourCustomModule: count_your_custom_module})

print(f"Custom MACs: {macs}, Parameters: {params}")
```

如果需要一个可读带单位的输出，如下所示，使用clever_format

```python3
macs_readable, params_readable = clever_format([macs, params], "%.3f")

print(f"Formatted MACs: {macs_readable}, Formatted Parameters: {params_readable}")
# Expected output: Formatted MACs: 4.140G, Formatted Parameters: 25.557M
```

