import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = torch.device("cuda:0")

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
print("sm number: ", NUM_SM, "\n register number: ", NUM_REGS, "\n shared memory size: ", SIZE_SMEM, "\n warp size: ", WARP_SIZE)