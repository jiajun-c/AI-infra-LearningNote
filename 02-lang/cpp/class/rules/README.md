# Rule of three (C98)

指的是在类中有三个函数是息息相关的
- 析构函数
- 拷贝构造函数
- 拷贝赋值运算符

假设我们只实现了析构函数但是没有实现拷贝构造和拷贝赋值函数，那么将会调用默认的移动构造函数，直接拷贝指针，在析构函数调用的时候将会发生double free的问题

如下所示

```cpp
#include <iostream>
#include <vector>

using namespace std;

class Myarray {
public:
    int *data;
    size_t size = 10;
    Myarray() {
        data = new int[10];
    }
    ~Myarray() {
        delete[] data;
    }
};


int main() {
    Myarray a;
    Myarray b = a;
}
```

一个正确的代码如下所示

```cpp
#include <iostream>
#include <vector>

using namespace std;

class Myarray {
public:
    int *data;
    size_t size = 10;
    Myarray() {
        data = new int[10];
    }
    ~Myarray() {
        delete[] data;
    }
    Myarray(const Myarray& arr) {
        printf("copy construct\n");
        data = new int[10];
        copy(arr.data, arr.data+10, data);
    }

    Myarray& operator=(const Myarray& arr) {
        printf("copy construct operator\n");
        data = new int[10];
        copy(arr.data, arr.data+10, data);
        return *this;
    }
};

int main() {
    Myarray a;
    Myarray b = a;
    Myarray c;
    c = a;
}
```

# Rule of five(C++11)

到了 C++11，移动语义（Move Semantics）横空出世，又加了两个新成员：
- 移动构造函数 (Move Constructor)
- 移动赋值操作符 (Move-Assignment)

代码如下所示，当我们没有指定移动构造函数和移动赋值操作符时他会自动fallback到拷贝构造上，如下所示会打印 `copy construct`

```cpp
#include <iostream>
#include <vector>

using namespace std;

class Myarray {
public:
    int *data;
    size_t size = 10;
    Myarray() {
        data = new int[10];
    }
    ~Myarray() {
        delete[] data;
    }
    Myarray(const Myarray& arr) {
        printf("copy construct\n");
        data = new int[10];
        copy(arr.data, arr.data+10, data);
    }

    Myarray& operator=(const Myarray& arr) {
        printf("copy construct operator\n");
        data = new int[10];
        copy(arr.data, arr.data+10, data);
        return *this;
    }
};


int main() {
    Myarray a;
    Myarray b = std::move(a);
}
```

补充移动语义函数如下所示

```cpp
    Myarray(const Myarray& arr) {
        printf("copy construct\n");
        data = new int[10];
        copy(arr.data, arr.data+10, data);
    }

    Myarray& operator=(const Myarray& arr) {
        printf("copy construct operator\n");
        data = new int[10];
        copy(arr.data, arr.data+10, data);
        return *this;
    }
```