import dataclasses
from typing import Callable, Optional, Tuple

import torch
from timm import bits
from timm.data.dataset_factory import create_dataset
from torch import nn
from torchvision import datasets

from attacks import AttackFn, TRADESLoss

_DATASETS = {'cifar10': datasets.CIFAR10, 'cifar100': datasets.CIFAR100}


def my_create_dataset(name: str,
                      root: str,
                      split: str = 'validation',
                      search_split: bool = True,
                      is_training: bool = False,
                      batch_size: Optional[int] = None,
                      **kwargs):
    """Wraps timm's create dataset to use also PyTorch's datasets"""
    name = name.lower()
    if not name.startswith('torch'):
        return create_dataset(name, root, split, search_split, is_training,
                              batch_size, **kwargs)

    dataset_name = name.split('/')[-1]
    return _DATASETS[dataset_name](root, train=is_training, download=True)


@dataclasses.dataclass
class AdvTrainState(bits.TrainState):
    compute_loss_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], Tuple[
        torch.Tensor, torch.Tensor,
        Optional[torch.Tensor]]] = None  # type: ignore

    @classmethod
    def from_bits(cls, instance: bits.TrainState, **kwargs):
        return cls(**dataclasses.asdict(instance), **kwargs)
