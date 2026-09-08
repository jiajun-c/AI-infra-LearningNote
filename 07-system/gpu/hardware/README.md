# GPU 硬件名词

## 1. 芯片代号

NVIDIA GPU 每一代有一个字母代号（SM 大版本）：

|代号|全称|代表产品|SM 版本|
|---|---|---|---|
|GV|Volta|V100|SM 7.0|
|TU|Turing|T4, RTX 20 系列|SM 7.5|
|GA|Ampere|A100, RTX 30 系列|SM 8.0 / 8.6|
|AD|Ada Lovelace|L40, RTX 40 系列|SM 8.9|
|GH|Grace Hopper|H100, H200|SM 9.0|
|GB|Grace Blackwell|B100, B200, GB200|SM 10.0|

`100` 通常是数据中心满血 die（如 GA100、GH100、GB100），`02` 等是消费级阉割版（如 GA102 = RTX 3090，AD102 = RTX 4090，GB202 = RTX 5090）。

## 2. 卡 / 模组形态（SKU 家族）

|缩写|全称|说明|
|---|---|---|
|PIB|PCIe Interface Board|标准 PCIe 卡，无 NVLink|
|PNB|PCIe NVL Board|PCIe 形态，板载 NVLink Switch 连接器|
|HGX|(NVIDIA 内部名)|8-GPU SXM baseboard|
|MGX|Modular GPU eXpress|模块化 GPU 服务器规范|
|NVL72|NVLink 72|整机柜级 NVLink 域（72 张 GPU）|
|GB200|Grace + 2×Blackwell|Grace CPU + 2 颗 Blackwell die 的超级芯片|
|DGX|Deep Learning GPU eXpress|NVIDIA 自家参考服务器整机|

### 2.1 PIB vs PNB 关键差异

|维度|PIB（PCIe Interface Board）|PNB（PCIe NVL Board）|
|---|---|---|
|接口形态|标准 PCIe 5.0 x16|PCIe 5.0 x16 + 板载 NVLink Switch 连接器|
|单卡互联|✗ 跨卡走 PCIe|✔ 走 NVLink Switch（每卡 600~900 GB/s）|
|目标部署|单台 8-GPU 服务器|NVL72 / 整机柜级 NVLink 域|
|典型 TDP|约 300~400 W|约 600~700 W（更高性能）|
|HBM 配置|通常标配|往往更高带宽/容量版本|

## 3. NVSwitch 各代规格

|代|GPU|单口带宽|端口数|单 Switch 总带宽|
|---|---|---|---|---|
|NVSwitch 1|Volta V100|约 25 GB/s|约 8|约 200 GB/s|
|NVSwitch 2|Ampere A100|约 50 GB/s|18|约 900 GB/s|
|NVSwitch 3|Hopper H100|50 GB/s|18|900 GB/s|
|NVSwitch 4|Blackwell B100/B200|100 GB/s（×2）|64（×3.5）|6400 GB/s（×7）|

**Hopper 时代**：18 ports × 50 GB/s = 900 GB/s 是干净的设计参数，NVLink 4 链路数与 NVSwitch 3 端口数完全对齐（每张 GPU 18 条 NVLink）。

**Blackwell 时代**：单口翻倍、端口数大幅扩展，是为支撑 NVL72 这种"72 张卡仍像 1 张卡一样互联"的扩展性。

## 4. NVLink 各代带宽

|代|GPU|单口带宽|单 GPU 链路数|单 GPU 总带宽|
|---|---|---|---|---|
|NVLink 3|Ampere A100|约 50 GB/s|12|600 GB/s|
|NVLink 4|Hopper H100|50 GB/s|18|900 GB/s|
|NVLink 5|Blackwell B100/B200|100 GB/s（×2）|18|1800 GB/s（×2）|

## 5. NVL72 整机柜架构

NVL72 把"1 rack = 1 系统"的理念落地：

```text
NVL72 整机柜
│
├── Compute 区
│   ├── Compute Tray 0  : 4 GPU + 1 Grace CPU
│   ├── Compute Tray 1  : 4 GPU + 1 Grace CPU
│   ├── ...
│   └── Compute Tray 17 : 4 GPU + 1 Grace CPU
│       共 18 compute tray = 72 GPU + 18 Grace
│
├── Switch 区
│   ├── Switch Tray 0   : 多颗 NVSwitch 4
│   ├── ...
│   └── Switch Tray N   : 多颗 NVSwitch 4
│
└── 整机柜视为一个 NVLink 域：
    - 任何两张 GPU 之间的通信都走 NVLink Switch
    - 跨 tray、跨 switch tray 全部在 NVLink 内部
    - 不需要 IB / RoCE
```

### 5.1 NVL72 整机柜带宽账

```text
单 GPU NVLink 5 总带宽：    1800 GB/s
72 张 GPU 聚合带宽：         72 × 1800 = 129.6 TB/s（双向 ~260 TB/s）

NVSwitch 4 单芯片总带宽：   6400 GB/s
要承担 129.6 TB/s 聚合，
需要 NVSwitch 4 数量 ≈ 129600 / 6400 ≈ 20 颗（粗估）

但 NVL72 不只满足"够用"，还要支持
  ├─ 任意 GPU ↔ 任意 GPU 全互联（all-to-all）
  ├─ 跨 tray 直连
  └─ 留出多级交换冗余
所以实际芯片数 ≥ 20 颗，分摊到多个 switch tray。
```

### 5.2 "18" 在 NVL72 里的多重含义

|指向|含义|
|---|---|
|18 compute trays|72/4 = 18，每 tray 4 GPU|
|18 NVLink links per GPU|每张 GPU 有 18 条 NVLink 链路|
|18 ports per NVSwitch|Hopper NVSwitch 3 的端口数|

注意：**不是 18 颗 NVSwitch**——NVL72 的 NVSwitch 4 芯片总数远超 18。

## 6. 集群层级与 NODE 概念

### 6.1 标准数据中心层级

```text
数据中心 (Data Center)
└── 集群 (Cluster)              ← 跨多机柜，IB/RoCE 互联
    └── 整机柜 (Rack)           ← 42U 机柜，约 10~20 台服务器
        └── 节点 (NODE/Server)  ← 一台物理服务器，含 N 张 GPU
            └── Baseboard       ← GPU 基板（HGX 等）
                └── GPU          ← 单卡
```

### 6.2 NODE 的多种含义

|语境|NODE = ?|包含的 GPU 数|
|---|---|---|
|MPI / 集群|一台物理服务器|通常 8 (HGX/DGX)|
|CUDA 编程|一个 host|通常 8|
|DGX SuperPOD 文档|一台 DGX 服务器|8|
|NVL72 拓扑 / 故障域文档|compute tray|4 GPU + 1 Grace CPU|
|数据中心物理|一个机柜（rack）|8 × N 台服务器|

**默认含义**：NODE = 一台服务器（HGX 8 卡 / DGX 整机）。  
**NVL72 例外**：文档里有时把 compute tray（4 卡 + Grace CPU）叫 NODE。

### 6.3 Rack ↔ NODE ↔ GPU 关系

|形态|Rack : NODE : GPU 关系|
|---|---|
|传统 PCIe 服务器集群|1 rack ⊃ N node : 8 GPU/node|
|HGX H100 SuperPOD|1 rack ⊃ N node : 8 GPU/node（每 node 一台 DGX）|
|NVL72|1 rack = 1 sys ⊃ 18 node : 4 GPU/node|
|OCP OAM 大型集群|1 rack ⊃ N node : 8 GPU/node|
|DGX SuperPOD（多机柜）|多个 rack ⊃ 多个 node : 8 GPU/node，靠 IB 跨 rack|

## 7. 通信相关名词

|名词|含义|
|---|---|
|NODE|单机 8 卡（标准），或 compute tray（NVL72 语境）|
|SYS NV18|由 18 颗 NVSwitch 构成的 NVLink 交换域（最常见于 NVL72 类大系统）|
|Rack|整机柜；传统集群里装多个 NODE；NVL72 里 = 整个系统|

### 7.1 Rack 与 NODE 的关系

传统 Rack：

```text
传统 Rack                    NVL72 Rack
┌────────────────┐           ┌────────────────┐
│ Server (8 GPU) │           │   Compute Tray │
│ Server (8 GPU) │           │   4 GPU + 1 Grace │
│ Server (8 GPU) │           │   Compute Tray │
│ ...            │           │   4 GPU + 1 Grace │
│ (各自独立)     │           │   ...           │
└────────────────┘           │   Switch Tray   │
                            │   Switch Tray   │
                            │   (全互联)      │
                            └────────────────┘
                            一个 rack = 一个系统
```

|维度|传统集群 Rack|NVL72 Rack|
|---|---|---|
|角色|只是物理外壳|一个完整的 NVLink 系统|
|内部 GPU 互联|各自独立，跨机走 IB|全 72 张在一个 NVLink 域|
|Switch 单元|每板自带 NVSwitch|整柜共享 NVLink Switch 域|
|NODE 含义|一台服务器|一个 compute tray (4 GPU)|
|Rack 与 NODE 关系|1 rack ⊃ N node|1 rack = 1 sys ⊃ 18 node|

### 7.2 整体集群拓扑示意

```text
DGX SuperPOD（多 rack）           NVL72（单 rack = 单系统）
─────────────────────            ────────────────────────
┌─ Rack 1 ─┐                     ┌──── Rack (整机柜) ────┐
│ Node A   │ 8 GPU               │ Node 0   │ 4 GPU     │
│ Node B   │ 8 GPU               │ Node 1   │ 4 GPU     │
│ Node C   │ 8 GPU               │   ...    │           │
└──────────┘                     │ Node 17  │ 4 GPU     │
┌─ Rack 2 ─┐                     │ ┌──────────────────┐ │
│ Node D   │ 8 GPU               │ │ Switch Trays     │ │
│ Node E   │ 8 GPU               │ │ (NVSwitch 4 ×N)  │ │
│ Node F   │ 8 GPU               │ └──────────────────┘ │
└──────────┘                     └──────────────────────┘
       │                                  │
   IB/RoCE 跨 rack               全部在 NVLink 内部
       │                                  │
   多个 NVLink 域               一个 NVLink 域
   (每 node 一个)              (整柜共享)
```

## 8. 互联总线

|概念|含义|典型带宽|
|---|---|---|
|PCIe|CPU ↔ GPU 标准接口（5.0 = 128 GB/s）|~128 GB/s|
|NVLink|GPU ↔ GPU 直连（H100: 900 GB/s/卡）|600/900/1800 GB/s 每代|
|NVSwitch|NVLink 的交换芯片，让多卡全互联|见 §3|
|NVLink-C2C|Grace ↔ Hopper/Blackwell 的一致性互连|900 GB/s|
|InfiniBand / RoCE|跨节点 GPU 互联（DGX SuperPOD）|~400 Gb/s / port|
