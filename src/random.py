import random
import numpy as np
import torch


def random_seed(seed=42, rank=0, dev_env=None):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    if dev_env is not None and dev_env.type_xla:
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(seed + rank, device=dev_env.device)