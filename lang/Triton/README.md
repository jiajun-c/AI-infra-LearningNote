# Triton 

## 1. 原理

Triton本质上是一种MLIR，其层级为

> triton-lang -> triton ir -> triton gpu dialect ->llvmir -> ptx -> SASS

在GPU上层级分为线程层级，warp层级，block层级，grid层级

Triton所面向的block层级，通过program_id去对应到特定的block，内部warp层级则是由编译器进行一个自动映射


