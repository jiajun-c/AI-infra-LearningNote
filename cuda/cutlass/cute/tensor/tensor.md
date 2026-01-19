# cute Tensor

## 1. Tensor的创建

当我们完成shape和layout的创建后，使用从外部传递的内存地址，可以开始进行tensor的创建，如下所示使用`make_tensor`创建一个global memory 上的tensor对象

```cpp
__global__ void create_tensor_kernel(float* global_ptr) {
    auto layout = make_layout(make_shape(2, 2, 2));
    auto tensor = make_tensor(make_gmem_ptr(global_ptr), layout);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // 使用 CuTe 内置的 print 函数
        printf("Layout info:\n");
        print(layout); 
        printf("\n\nTensor info:\n");
        print(tensor);
        printf("\n");
        
        // 演示：访问 Tensor 的第 0 个元素
        // 语法: tensor(坐标)
        printf("Element at index 0: %f\n", tensor(0, 1, 1));
    }
}
```

创建共享内存上的Tensor如下所示，使用`make_smem_ptr`创建一个共享内存上的指针，然后使用`make_tensor`创建一个共享内存上的tensor对象

```cpp
__shared__ float smem[64];
auto smemLayout = make_layout(make_shape(4, 4, 4));
auto tensor_smem = make_tensor(make_smem_ptr(smem), smemLayout);
```

## 2. Tensor的访问

tensor的访问的访问方式有下面的几种
- `[]` 仅一个值
- `()` 可以传递多个值
- `make_coord` 更直观地访问多维度地Tensor