# NVLink 

这里主要是介绍NVLink和NVSwitch的硬件

## 1. 基本概念

NVSwitch：NVIDIA 推出的高速互联芯片

NVLink：NVSwitch在实现高速互联时的数据通路

## 2. NVSwitch

NVSwitch的作用相当于一个数据的中转站，可以高效地将数据传递到不同的GPU上

- 在没有NVSwitch的情况下，虽然GPU0有六条数据通路，但是其无法将其全部用于向GPU1的数据传输，因为这些数据通路中可能只有一条是指向GPU1的。
- 引入NVSwitch后，相当于构建了一个超级枢纽，每个GPU都连接到NVSwitch上，这意味着在GPU0向GPU1进行数据传递的时候可以充分利用上全部的NVLink通路


