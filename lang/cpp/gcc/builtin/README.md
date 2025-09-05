# gcc builtin 函数

## popcnt

popcnt 函数用于计算一个二进制数下数字中1的数目。

针对不同类型有下面的三种

```cpp
int __builtin_popcount (unsigned int)
int __builtin_popcountl (unsigned long)
int __builtin_popcountll (unsigned long long)
```

```cpp
#include <iostream>

using namespace std;

int main()
{
    int x = -1;
    printf("%d\n", __builtin_popcount(x));
}
```