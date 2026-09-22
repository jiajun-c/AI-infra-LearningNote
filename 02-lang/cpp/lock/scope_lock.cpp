#include <iostream>
#include <mutex>
#include <vector>
#include <thread>
#include <string>

using namespace std;

struct Account
{
    /* data */
    std::mutex m;
    long balance;
};


void transfer(Account& from, Account& to, long amount) {                                 
    std::scoped_lock lock(from.m, to.m);   // 不是 from.m.lock() 再 to.m.lock()          
    if (from.balance < amount) return;                                                   
    from.balance -= amount;                                                              
    to.balance += amount;                                                                
}

int main() {
    Account a{{}, 1000};
    Account b{{}, 1000};
    std::thread t1([&]{
        transfer(a, b, 100);
    });
    std::thread t2([&]{
        transfer(b, a, 150);
    });
    t1.join();
    t2.join();
    printf("%d %d\n",a.balance, b.balance);
}
