# torch context 机制

torch中的context是一个全局的单例，如下所示，每个进程只拥有一个context对象

```cpp
// Context.cpp:96-99
Context& globalContext() {
  static Context globalContext_;  // C++11 线程安全的静态局部变量
  return globalContext_;
}
```

里面存放的是一些配置信息，比如使用cudnn之类的

