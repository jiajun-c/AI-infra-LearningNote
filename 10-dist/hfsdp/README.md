# HFSDP

Hyprid FSDP，是一种混合FSDP策略，其分为两层，在外层DP采用zero1的形式只切分优化器的状态，然后在内层DP采用zero3的形式切分全部的状态
