# unordered 结构

unordered_map内部本质是一个桶数组。

```cpp
bucket 0: -> element A
bucket 1: -> element B -> element C
bucket 2: empty
bucket 3: -> element D
```

插入键值对的时候，不同的的键会被映射到同一个桶中，一个桶中可能有多个元素

## rehash

使用`bucket_count`可以得到map的bucket数量，`load_factor`可以得到负载因子

```cpp
#include <iostream>
#include <unordered_map>

int main() {
    std::unordered_map<int, int> map;

    std::size_t previous = map.bucket_count();

    for (int i = 0; i < 100; ++i) {
        map.emplace(i, i);

        if (map.bucket_count() != previous) {
            std::cout
                << "size = " << map.size()
                << ", bucket_count: "
                << previous << " -> "
                << map.bucket_count()
                << ", load_factor = "
                << map.load_factor()
                << '\n';

            previous = map.bucket_count();
        }
    }
}
```

进行rehash的时候，可以建立每个元素所属的通信，重建桶之间的组织关系。所以这种时候迭代器失效，元素引用和指针保持有效。
