#include <iostream>

using namespace std;

struct ListNode
{
    ListNode *_next;
    ListNode *_prev;
    int _data;

    ListNode(int val)
        :_next(nullptr)
        , _prev(nullptr)
        , _data(val)
    {

    }
    void* operator new(size_t n) {
        void *p = nullptr;
        p = allocator<ListNode>().allocate(1);
        cout << "memory pool allocate ListNode" << endl;
        return p;    
    }

    void operator delete(void *p) {
        allocator<ListNode>().deallocate((ListNode*)p, 1);
        cout << "memory pool delete ListNode" << endl;
    }
};
class List
{
public:
    List()
    {
        _head = new ListNode(-1);
        _head->_next = _head;
        _head->_prev = _head;
    }

    void PushBack(int val)
    {
        ListNode* newnode = new ListNode(val);
        ListNode* tail = _head->_prev;

        tail->_next = newnode;
        newnode->_prev = tail;
        newnode->_next = _head;
        _head->_prev = newnode;
    }

    ~List()
    {
        ListNode* cur = _head->_next;
        while (cur != _head)
        {
            ListNode* next = cur->_next;
            delete cur;
            cur = next;
        }

        delete _head;
        _head = nullptr;
    }

private:
    ListNode* _head;
};
int main() {
    List l;
    l.PushBack(1);
    l.PushBack(2);
    l.PushBack(3);
    l.PushBack(4);
    l.PushBack(5);

    return 0;
}