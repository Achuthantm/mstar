import os
from datetime import timedelta

import torch
import torch.distributed as dist

rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=timedelta(seconds=30), device_id=torch.device('cuda',rank))
x=torch.tensor([rank+1.],device=f'cuda:{rank}')
dist.all_reduce(x)
torch.cuda.synchronize()
assert x.item()==3., x
print(f'PASS rank={rank} all_reduce={x.item()}',flush=True)
dist.destroy_process_group()
