# 各种并行方式的组合对显存的影响

## 1. 限制

N = PPxTPxCPxDP = PPxETPxEPxEDP

## 2. 显存情况

- dense参数量：$P_d$
- Expert参数量:$P_e$
- Dense TP: T
- Dense DP: $D_d$ 
- EP: P
- ETP: $T_e$
- EDP: $D_e$

### 2.1 EP+TP

$P_{GPU} = \frac{P_d}{T} + \frac{P_e}{PT_e}$

### 2.2 EP+TP+DP

每个本地模型分片参数

$N_d = \frac{P_d}{T}$

$N_e = \frac{P_e}{P T_e}$

不使用ZeRO的时候 $M_{GPU} = 16(N_e + N_d)$

使用ZeRO1的时候 $M_{GPU} = (4 + \frac{12}{D_d})N_d + (4+\frac{12}{D_e})N_e$

