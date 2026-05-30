// cfs_demo.cpp
#include <cerrno>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sched.h>
#include <unistd.h>
#include <vector>

struct Child {
    std::string name;
    int nice_value;
    pid_t pid;
};


void bind_to_cpu0() {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(0, &set);

    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        std::cerr << "sched_setaffinity failed: " << strerror(errno) << "\n";
        _exit(1);
    }
}

void busy_loop(int nice_value) {
    if (setpriority(PRIO_PROCESS, 0, nice_value) != 0) {
        std::cerr << "setpriority failed: " << strerror(errno) << "\n";
        _exit(1);
    }

    bind_to_cpu0();

    volatile unsigned long long x = 0;
    while (true) {
        ++x;
    }
}


double read_cpu_time(pid_t pid) {
    std::ifstream file("/proc/" + std::to_string(pid) + "/stat");
    if (!file) {
        throw std::runtime_error("failed to open /proc/<pid>/stat");
    }

    std::string line;
    std::getline(file, line);

    std::istringstream iss(line);
    std::string field;

    long utime = 0;
    long stime = 0;

    for (int i = 1; iss >> field; ++i) {
        if (i == 14) {
            utime = std::stol(field);
        } else if (i == 15) {
            stime = std::stol(field);
            break;
        }
    }

    long ticks = sysconf(_SC_CLK_TCK);
    return static_cast<double>(utime + stime) / ticks;
}

int main() {
    std::vector<Child> children;

    std::vector<std::pair<std::string, int>> configs = {
        {"normal", 0},
        {"low-priority", 10},
    };

    for (const auto& [name, nice_value] : configs) {
        pid_t pid = fork();

        if (pid < 0) {
            std::cerr << "fork failed: " << strerror(errno) << "\n";
            return 1;
        }

        if (pid == 0) {
            busy_loop(nice_value);
        }

        children.push_back({name, nice_value, pid});
    }

    sleep(1);

    std::vector<double> start;
    for (const auto& child : children) {
        start.push_back(read_cpu_time(child.pid));
    }

    int duration = 10;
    std::cout << "run " << duration << "s...\n\n";
    sleep(duration);

    std::vector<double> end;
    for (const auto& child : children) {
        end.push_back(read_cpu_time(child.pid));
    }

    for (size_t i = 0; i < children.size(); ++i) {
        double cpu_time = end[i] - start[i];

        std::cout << children[i].name
                  << " nice=" << children[i].nice_value
                  << " pid=" << children[i].pid
                  << " cpu_time=" << cpu_time << "s\n";
    }

    for (const auto& child : children) {
        kill(child.pid, SIGKILL);
        waitpid(child.pid, nullptr, 0);
    }

    return 0;
}
