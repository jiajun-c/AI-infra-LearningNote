# ARM SME

## 1. 架构

ARM SME是基于ARM SVE拓展上实现的，其功能包括
- 计算SVE向量的外积
- 矩阵块(tile)存储
- tile向量的加载，存储，插入和提取
- StreamingSVE 模型

### 1.1 Streaming 模式

ARM SME中的指令大多需要在Streaming mode下进行使用。

在函数开头加上`__arm_locally_streaming`可以开启流模式，离开函数后将会结束

在streaming mode下的流向量长度与非streaming模式下的向量长度不同，SME的SVL可以是128，256，512，1024，2048bits，而NSVL需要是128的整数倍

```cpp
#include <stdio.h>
#include <arm_sme.h>

__arm_locally_streaming
void streaming_fn(void)
{
    printf("Streaming mode: svcntw() = %u, svcntsw() = %u swcnth()=%u\n", (unsigned)svcntw(), (unsigned)svcntsw(),(unsigned)svcnth());
}

int main(void)
{
    printf("Has SME? %s\n", __arm_has_sme() ? "Yes" : "No");
    printf("Non-streaming mode: svcntw() = %u, svcntsw() = %u swcnth()=%u\n", (unsigned)svcntw(), (unsigned)svcntsw(), (unsigned)svcnth());
    streaming_fn();
}
```

根据文档，在流模式下svcntw和swcntgw相同，上面代码的预期结果也符合
 
```c
// Return the number of words in a streaming vector.
// Equivalent to svcntw() when called in streaming mode.
uint64_t svcntsw() __arm_streaming_compatible;
```

### 1.2 ZA寄存器

SME中指令的实现是基于一个二维的ZA寄存器，其大小是SVLxSVL，对于ZA array可以通过如下的几种方式进行访问
- ZA array vector访问
- ZA tiles
- ZA tiles slices

## 2. 指令

`svzero_za` 清空ZA寄存器


`mopa` 类命令是sme中中的核心命令，其用于外积的计算。