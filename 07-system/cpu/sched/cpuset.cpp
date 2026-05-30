#include <sched.h>
#include <unistd.h>
#include <iostream>

int main() {
    cpu_set_t set;

    CPU_ZERO(&set);
    CPU_SET(0, &set);
    CPU_SET(1, &set);

    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        perror("sched_setaffinity");
        return 1;
    }

    std::cout << "current process can run on CPU 0 and CPU 1\n";
    return 0;
}