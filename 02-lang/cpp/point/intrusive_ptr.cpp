// 侵入式智能指针实现与 shared_ptr 性能对比
//
// 侵入式指针要求被管理的对象自己继承引用计数基类（RefCounted），
// 计数直接嵌入对象内存，省去了 shared_ptr 的独立控制块分配。
//
// 编译: g++ -O2 -std=c++17 intrusive_ptr.cpp -o intrusive_ptr
//
// 关键区别:
//   shared_ptr<T>(new T) : 两次 malloc（对象 + 控制块）
//   make_shared<T>       : 一次 malloc（对象 + 控制块合并）
//   intrusive_ptr<T>     : 一次 malloc（计数嵌入对象，无控制块）
//                          且对象可通过裸指针随时构造出 intrusive_ptr

#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <chrono>
#include <vector>

// ============================================================
// 侵入式引用计数基类
// ============================================================

class RefCounted {
public:
    RefCounted() : _ref(0) {}
    RefCounted(const RefCounted&) : _ref(0) {}  // 拷贝不继承计数
    RefCounted& operator=(const RefCounted&) { return *this; }

    void add_ref() noexcept {
        _ref.fetch_add(1, std::memory_order_relaxed);
    }

    // 返回 true 表示计数降到 0，调用者需负责销毁
    bool release() noexcept {
        return _ref.fetch_sub(1, std::memory_order_acq_rel) == 1;
    }

    long use_count() const noexcept {
        return _ref.load(std::memory_order_relaxed);
    }

private:
    mutable std::atomic<long> _ref;
};

// ============================================================
// intrusive_ptr 实现
// ============================================================

template<typename T>
class intrusive_ptr {
public:
    using element_type = T;

    constexpr intrusive_ptr() noexcept : _p(nullptr) {}

    explicit intrusive_ptr(T* p, bool add_ref = true) noexcept : _p(p) {
        if (_p && add_ref) _p->add_ref();
    }

    intrusive_ptr(const intrusive_ptr& o) noexcept : _p(o._p) {
        if (_p) _p->add_ref();
    }

    intrusive_ptr(intrusive_ptr&& o) noexcept : _p(o._p) {
        o._p = nullptr;
    }

    template<typename U>
    intrusive_ptr(const intrusive_ptr<U>& o) noexcept : _p(o.get()) {
        if (_p) _p->add_ref();
    }

    ~intrusive_ptr() {
        if (_p && _p->release()) delete _p;
    }

    intrusive_ptr& operator=(const intrusive_ptr& o) noexcept {
        intrusive_ptr tmp(o);
        swap(tmp);
        return *this;
    }

    intrusive_ptr& operator=(intrusive_ptr&& o) noexcept {
        intrusive_ptr tmp(std::move(o));
        swap(tmp);
        return *this;
    }

    T* get() const noexcept { return _p; }
    T& operator*()  const noexcept { return *_p; }
    T* operator->() const noexcept { return _p; }
    explicit operator bool() const noexcept { return _p != nullptr; }

    long use_count() const noexcept {
        return _p ? _p->use_count() : 0;
    }

    void reset(T* p = nullptr, bool add_ref = true) {
        intrusive_ptr tmp(p, add_ref);
        swap(tmp);
    }

    void swap(intrusive_ptr& o) noexcept { std::swap(_p, o._p); }

private:
    T* _p;
};

template<typename T, typename... Args>
intrusive_ptr<T> make_intrusive(Args&&... args) {
    // add_ref=false: 构造函数里已经是 new，不重复 add_ref
    return intrusive_ptr<T>(new T(std::forward<Args>(args)...), true);
}

// ============================================================
// 被管理的测试对象
// ============================================================

struct Node : public RefCounted {
    int value;
    char padding[60];   // 凑 64 字节，与 shared_ptr 版对比时大小相近
    explicit Node(int v) : value(v) { memset(padding, 0, sizeof(padding)); }
};

struct NodePlain {
    int value;
    char padding[60];
    explicit NodePlain(int v) : value(v) { memset(padding, 0, sizeof(padding)); }
};

// ============================================================
// 模拟"对象自带引用计数"的场景
// 典型例子：COM IUnknown、WebKit RefCounted、Android sp<>
//
// 对象自己管理计数，外部代码通过 AddRef/Release 接管生命周期，
// intrusive_ptr 可以零成本包装裸指针；
// shared_ptr 则必须额外创建控制块（enable_shared_from_this 也一样，
// 它只是把控制块的创建推迟到第一次构造 shared_ptr 时）。
// ============================================================

// 模拟 COM-style 对象：自带计数，提供 AddRef/Release 接口
class ComLikeObject {
public:
    explicit ComLikeObject(int v) : value(v), _ref(1) {
        memset(padding, 0, sizeof(padding));
    }

    // COM 风格接口
    void AddRef() noexcept {
        _ref.fetch_add(1, std::memory_order_relaxed);
    }
    void Release() noexcept {
        if (_ref.fetch_sub(1, std::memory_order_acq_rel) == 1)
            delete this;
    }
    long RefCount() const noexcept {
        return _ref.load(std::memory_order_relaxed);
    }

    int value;
    char padding[52];

private:
    std::atomic<long> _ref;  // 嵌入对象内部，不需要外部控制块

    ~ComLikeObject() = default;  // 只能通过 Release 销毁
};

// 为 ComLikeObject 适配 intrusive_ptr 所需的 add_ref/release 接口
// 这是侵入式指针的标准做法：通过 ADL 或 traits 解耦
inline void intrusive_ptr_add_ref(ComLikeObject* p) noexcept { p->AddRef(); }
inline void intrusive_ptr_release(ComLikeObject* p) noexcept { p->Release(); }

// intrusive_ptr 的通用版本：通过 ADL 找 add_ref/release，
// 不强制要求继承 RefCounted
template<typename T>
class intrusive_ptr_adl {
public:
    using element_type = T;

    constexpr intrusive_ptr_adl() noexcept : _p(nullptr) {}

    // add_ref=false 用于接管已持有引用的裸指针（如 COM GetObject 返回 ref+1）
    explicit intrusive_ptr_adl(T* p, bool add_ref = true) noexcept : _p(p) {
        if (_p && add_ref) intrusive_ptr_add_ref(_p);
    }

    intrusive_ptr_adl(const intrusive_ptr_adl& o) noexcept : _p(o._p) {
        if (_p) intrusive_ptr_add_ref(_p);
    }

    intrusive_ptr_adl(intrusive_ptr_adl&& o) noexcept : _p(o._p) {
        o._p = nullptr;
    }

    ~intrusive_ptr_adl() {
        if (_p) intrusive_ptr_release(_p);
    }

    intrusive_ptr_adl& operator=(const intrusive_ptr_adl& o) noexcept {
        intrusive_ptr_adl tmp(o);
        std::swap(_p, tmp._p);
        return *this;
    }

    intrusive_ptr_adl& operator=(intrusive_ptr_adl&& o) noexcept {
        std::swap(_p, o._p);
        return *this;
    }

    T* get() const noexcept { return _p; }
    T& operator*()  const noexcept { return *_p; }
    T* operator->() const noexcept { return _p; }
    explicit operator bool() const noexcept { return _p != nullptr; }

private:
    T* _p;
};

// enable_shared_from_this 版本：shared_ptr 管理自带计数的对象
// 注意：这需要对象继承 enable_shared_from_this，且必须通过 shared_ptr 创建
class ComLikeShared : public std::enable_shared_from_this<ComLikeShared> {
public:
    explicit ComLikeShared(int v) : value(v) {
        memset(padding, 0, sizeof(padding));
    }
    int value;
    char padding[56];
};

// ============================================================
// 计时工具
// ============================================================

using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(Clock::time_point t0) {
    auto dt = Clock::now() - t0;
    return std::chrono::duration<double, std::milli>(dt).count();
}

// ============================================================
// 测试项
// ============================================================

static const int N = 10'000'000;

// 测试1: 大量创建/销毁
void bench_create_destroy() {
    printf("=== 创建/销毁 %d 次 ===\n", N);

    // intrusive_ptr
    {
        auto t0 = Clock::now();
        for (int i = 0; i < N; ++i) {
            auto p = make_intrusive<Node>(i);
            (void)p;
        }
        printf("intrusive_ptr make:          %.2f ms\n", elapsed_ms(t0));
    }

    // make_shared (单次分配)
    {
        auto t0 = Clock::now();
        for (int i = 0; i < N; ++i) {
            auto p = std::make_shared<NodePlain>(i);
            (void)p;
        }
        printf("make_shared:                 %.2f ms\n", elapsed_ms(t0));
    }

    // shared_ptr(new T) (两次分配)
    {
        auto t0 = Clock::now();
        for (int i = 0; i < N; ++i) {
            std::shared_ptr<NodePlain> p(new NodePlain(i));
            (void)p;
        }
        printf("shared_ptr(new T):           %.2f ms\n", elapsed_ms(t0));
    }
}

// 测试2: 拷贝（只涉及引用计数增减，不触发内存分配）
// 用 vector 存储拷贝防止被优化消除
void bench_copy() {
    printf("\n=== 拷贝/积累 %d 次（vec 存储防优化）===\n", N);

    {
        auto src = make_intrusive<Node>(1);
        std::vector<intrusive_ptr<Node>> vec;
        vec.reserve(N);
        auto t0 = Clock::now();
        for (int i = 0; i < N; ++i) {
            vec.push_back(src);
        }
        printf("intrusive_ptr copy (use=%ld): %.2f ms\n",
               src.use_count(), elapsed_ms(t0));
    }

    {
        auto src = std::make_shared<NodePlain>(1);
        std::vector<std::shared_ptr<NodePlain>> vec;
        vec.reserve(N);
        auto t0 = Clock::now();
        for (int i = 0; i < N; ++i) {
            vec.push_back(src);
        }
        printf("shared_ptr copy   (use=%ld): %.2f ms\n",
               src.use_count(), elapsed_ms(t0));
    }
}

// 测试3: 通过裸指针低成本重新包装（侵入式指针的独特优势）
void bench_wrap_raw() {
    printf("\n=== 裸指针重新包装 %d 次（侵入式优势场景）===\n", N);

    // 模拟已有裸指针池（如 C API 返回的指针）
    Node* raw = new Node(42);
    raw->add_ref();  // 初始持有

    {
        auto t0 = Clock::now();
        for (int i = 0; i < N; ++i) {
            // 从裸指针构造 intrusive_ptr 不需要额外内存，O(1) add_ref
            intrusive_ptr<Node> p(raw);
            (void)p;
        }
        printf("intrusive_ptr from raw:      %.2f ms\n", elapsed_ms(t0));
    }

    // shared_ptr 无法从裸指针安全重新包装（每次都创建新控制块）
    // 只能用 enable_shared_from_this，但有限制
    printf("shared_ptr from raw:         N/A (需 enable_shared_from_this)\n");

    raw->release();  // 释放初始引用
    delete raw;
}

// 测试4: 模拟接管"已有内建计数的对象"（COM/WebKit 场景）
//
// 场景：某工厂函数返回 ComLikeObject*，引用计数已经是 1。
//   intrusive_ptr_adl(raw, false)  -> 直接接管，不额外 AddRef，零开销
//   shared_ptr                     -> 必须额外分配控制块（~32 bytes heap）
//
// 这就是 shared_ptr 在此场景下的固有成本：即使对象有自己的计数，
// shared_ptr 也必须再维护一套独立计数，无法消除控制块分配。
void bench_preexisting_refcount() {
    printf("\n=== 接管已有计数对象 %d 次（COM/WebKit 典型场景）===\n", N);

    // --- intrusive_ptr_adl: 接管，不额外 AddRef ---
    {
        // 模拟工厂：每次返回 ref=1 的对象
        std::vector<ComLikeObject*> pool;
        pool.reserve(N);
        for (int i = 0; i < N; ++i)
            pool.push_back(new ComLikeObject(i));  // ref=1，已持有

        auto t0 = Clock::now();
        for (int i = 0; i < N; ++i) {
            // add_ref=false：接管工厂已给的那个引用，无额外操作
            intrusive_ptr_adl<ComLikeObject> p(pool[i], false);
            // p 析构时调用 Release，计数降到 0 自动 delete
        }
        printf("intrusive_ptr_adl(raw,false) 接管: %.2f ms  (0 额外 alloc)\n",
               elapsed_ms(t0));
        // pool 中的对象已被 intrusive_ptr_adl 析构时全部 Release 掉
    }

    // --- shared_ptr: 必须为每个对象额外分配控制块 ---
    {
        std::vector<ComLikeShared*> pool;
        pool.reserve(N);
        for (int i = 0; i < N; ++i)
            pool.push_back(new ComLikeShared(i));

        auto t0 = Clock::now();
        for (int i = 0; i < N; ++i) {
            // 每次都触发一次控制块 malloc，即使对象本身已有 enable_shared_from_this
            std::shared_ptr<ComLikeShared> p(pool[i]);
        }
        printf("shared_ptr(raw) 接管:              %.2f ms  (%d 次控制块 alloc)\n",
               elapsed_ms(t0), N);
    }

    // --- shared_ptr + make_shared: 避免额外 alloc，但对象必须由 make_shared 创建 ---
    {
        std::vector<std::shared_ptr<ComLikeShared>> pool;
        pool.reserve(N);
        for (int i = 0; i < N; ++i)
            pool.push_back(std::make_shared<ComLikeShared>(i));

        auto t0 = Clock::now();
        for (int i = 0; i < N; ++i) {
            // shared_from_this() 从已有控制块增加引用，不新建控制块
            auto p = pool[i]->shared_from_this();
            (void)p;
        }
        printf("shared_from_this() 增加引用:       %.2f ms  (0 额外 alloc，但必须 make_shared 创建)\n",
               elapsed_ms(t0));
    }
}

// ============================================================
// sizeof 展示
// ============================================================

void show_sizes() {
    printf("=== sizeof ===\n");
    printf("shared_ptr<NodePlain>:            %zu bytes (裸指针 + 控制块指针)\n",
           sizeof(std::shared_ptr<NodePlain>));
    printf("intrusive_ptr<Node>:              %zu bytes (只有裸指针)\n",
           sizeof(intrusive_ptr<Node>));
    printf("intrusive_ptr_adl<ComLikeObject>: %zu bytes (只有裸指针)\n",
           sizeof(intrusive_ptr_adl<ComLikeObject>));
    printf("NodePlain (无计数):               %zu bytes\n", sizeof(NodePlain));
    printf("Node      (继承 RefCounted):      %zu bytes (+atomic<long>)\n", sizeof(Node));
    printf("ComLikeObject (内建计数):         %zu bytes (+atomic<long>)\n", sizeof(ComLikeObject));
    printf("ComLikeShared (enable_shared):    %zu bytes (+weak_ptr _M_weak_this)\n", sizeof(ComLikeShared));
    printf("shared_ptr 控制块 (估算):         ~32 bytes (vptr+use+weak+ptr, 堆分配)\n");
}

int main() {
    show_sizes();
    bench_create_destroy();
    bench_copy();
    bench_wrap_raw();
    bench_preexisting_refcount();
    return 0;
}
