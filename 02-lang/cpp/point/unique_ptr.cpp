#include <iostream>
#include <memory>

struct Task {
    int mId;
    Task(int id ) :mId(id) {
        std::cout << "Task::Constructor" << std::endl;
    }
    ~Task() {
        std::cout << "Task::Destructor " << mId <<  std::endl;
    }
};

int main()
{
    // 通过原始指针创建 unique_ptr 实例
    std::unique_ptr<Task> uniqueTaskPtr(new Task(11));
    std::unique_ptr<Task> uniqueTaskPtr1 = std::move(uniqueTaskPtr);

    printf("uniqueTaskPtr.get() = %p\n", uniqueTaskPtr.get());
    auto taskPtr =  new Task(23);

    //通过 unique_ptr 访问其成员
    int id = taskPtr->mId;
    std::cout << id << std::endl;

    return 0;
}
