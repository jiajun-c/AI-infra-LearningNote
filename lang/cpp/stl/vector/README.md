#  Vector

## 1. vector类原理

vector类是动态数组，随着数据的加入，他的内部机制将会自动扩充来容纳新元素。

vector的内存管理由M_impl中的M_start, M_finish, M_end_of_storage三个指针来管理。所有关于地址，容量大小等操作都要用到这三个指针。

- M_start: 指向迭代器位置的指针
- M_finish: 指向迭代器结束位置的指针
- M_end_of_storage: 存储空间结束位置的指针

