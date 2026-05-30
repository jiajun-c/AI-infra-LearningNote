# cfs_demo.py
import os
import time
import signal
import resource

def busy_loop(name, nice_value):
    os.nice(nice_value)

    # 绑到同一个 CPU，避免它们跑到不同核上
    os.sched_setaffinity(0, {0})

    x = 0
    while True:
        x += 1

def read_cpu_time(pid):
    # /proc/<pid>/stat 里第 14、15 列是 utime/stime
    with open(f"/proc/{pid}/stat") as f:
        fields = f.read().split()

    utime = int(fields[13])
    stime = int(fields[14])
    ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])

    return (utime + stime) / ticks

children = []

for name, nice_value in [("normal", 0), ("low-priority", 10)]:
    pid = os.fork()
    if pid == 0:
        busy_loop(name, nice_value)
    else:
        children.append((name, nice_value, pid))

time.sleep(1)

start = {
    pid: read_cpu_time(pid)
    for _, _, pid in children
}

duration = 10
print(f"run {duration}s...\n")
time.sleep(duration)

end = {
    pid: read_cpu_time(pid)
    for _, _, pid in children
}

for name, nice_value, pid in children:
    cpu_time = end[pid] - start[pid]
    print(f"{name:12s} nice={nice_value:2d} pid={pid} cpu_time={cpu_time:.2f}s")

for _, _, pid in children:
    os.kill(pid, signal.SIGKILL)
