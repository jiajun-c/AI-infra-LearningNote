import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def allreduce_func(rank, size):
    group = dist.new_group(list(range(size)))
    tensor = torch.ones(1).to(torch.device("cpu", rank))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor)

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '9999'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def run(world_size, func):
    processes = []
    mp.set_start_method('spawn')
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, func))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

if __name__ == "__main__":
    run(2, allreduce_func)