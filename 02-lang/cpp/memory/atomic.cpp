#include <atomic>
#include <iostream>
#include <thread>
template <class T>
class RefCounted {
    std::atomic<int> cnt_{1};
    T* ptr_;
public:
    explicit RefCounted(T* p) : ptr_(p) {}

    // 访问底层指针，不增引用计数（仅借用，不接管所有权）
    T* get() const { return ptr_; }

    void addRef() {
        // 增加：relaxed 足够。因为持有引用本身就说明对象已经可见
        cnt_.fetch_add(1, std::memory_order_relaxed);
    }

    void release() {
        // 减少：必须 acq_rel
        //   release 半边 —— 本线程对对象的修改要对"最后那个人"可见
        //   acquire 半边 —— 本线程要看到别人此前的修改，才能安全析构
        if (cnt_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete ptr_;
        }
    }
};


int main() {
    // 1) 对象必须在堆上 —— release 最后一调用会 delete ptr_
    int* p1 = new int{21};

    // 2) 主线程先拿走第一份所有权，cnt = 1
    RefCounted<int> ref(p1);

    // 3) 想把同一份对象共享给 t1，必须先 addRef：cnt = 2
    //    否则 t1 一 release 就把对象删了，主线程后续访问就炸
    ref.addRef();

    std::thread t1([&ref]{
        // t1 通过引用共享同一个对象
        std::cout << "t1 sees: " << *ref.get() << std::endl;
        ref.release();          // t1 用完，cnt = 1
    });

    ref.release();              // 主线程用完，cnt = 0 → 自动 delete p1
    t1.join();                  // 等 t1 跑完
    return 0;
}