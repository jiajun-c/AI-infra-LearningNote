# GPU 拓扑相连

对于8卡的，其组织形式可能如下所示

```shell
节点 A：GPU0 GPU1 GPU2 GPU3 ─┐
                              ├─ 机柜内 NVLink / NVSwitch 网络
节点 B：GPU4 GPU5 GPU6 GPU7 ─┘
```