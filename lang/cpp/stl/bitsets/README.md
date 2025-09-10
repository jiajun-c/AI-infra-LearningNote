# Bitset 库使用

bitset 库类似一种二进制数组的形式，每一位的元素为0或者1，且只占用1bit的空间。

`bitset<n>` n表示的是bitset的元素数量。其提供了相应的操作接口


bitset 元素之间可以进行 `and`, `or`, `xor` 之类或者位运算的操作。

- `bitset.count()` 计算bitset中1的数量
- `bitset.any()` 判断bitset是否包含1
- `bitset.none()` 判断bitset是否全为0
- `bitset.set(index, 1)` 设置bitset的某一位为1, 如果不指定设置某一位，那么全部为1

```cpp
#include <iostream>
#include <bitset>

using namespace std;

int main()
{
    bitset<128>bitset1; 
    bitset<128>bitset2;
    
    bitset2.set(1, 1);
    printf("the bitset2 count: %d\n", bitset2.count());
    bitset1[0] = 1;
    printf("the bitset1 count: %d\n", bitset1.count());
    printf("the bitset1 has one: %d\n", bitset1.any());
    printf("the bitset1 has zero: %d\n", bitset1.none());
}
```