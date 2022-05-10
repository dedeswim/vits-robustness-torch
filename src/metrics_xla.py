import torch
import torch_xla.core.xla_model as xm
from timm.bits import AccuracyTopK, AvgTensor, DeviceEnv

class AvgTensorXLA(AvgTensor):
    def __init__(self, dev_env: DeviceEnv, accumulate_dtype=torch.float32):
        super().__init__(accumulate_dtype)
        self.dev_env = dev_env
    
    def compute(self):
        local_avg = super().compute()
        cctx = xm.CollectiveContext()
        avg = xm.all_reduce(xm.REDUCE_SUM, local_avg, scale=1.0 / self.dev_env.world_size, cctx=cctx)
        return avg
        