# linux 设备信息

## 1. BDF

BDF是PCI设备的地址，带上前面的domain后是DBDF

例如对于一个地址他是0012:03:00.0，0012是domain，表示pcie域，03表示总线编号，00表示该总线编号，0表示设备的功能编号

一个设备可能有多个Funcion，例如多端口网卡可能有多个pcie功能

但是对于`/sys/bus/pci/devices/0000:03:00:0`，其也仅是一个符号链接，其对应的实际地址通常为`/sys/devices/pci000.00/0000:00:01/0000:03:00.0`

可以理解为`devices+pci根总线+上游桥设备+地址`

通过这个地址可以判断两个设备的距离，是否是共享PCI总线

获取所在的numa

`/sys/bus/pci/deivces/bdf/numa_node` 可以获取对应的numa id

