import pysmctrl

# get_tpc_info_cuda 不需要 nvdebug 模块，直接使用 CUDA API
print("TPC count (via CUDA):", pysmctrl.get_tpc_info_cuda(0))

# 下面这两个函数需要 nvdebug 内核模块加载才能使用
# print("GPC info:", pysmctrl.get_gpc_info(0))
# print("TPC count:", pysmctrl.get_tpc_info(0))
