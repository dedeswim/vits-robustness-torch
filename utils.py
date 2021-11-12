import dataclasses
from typing import Callable, Optional, Tuple

import torch
from timm import bits
from timm.data.dataset_factory import create_dataset
from torch import nn
from torchvision import datasets

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


class ComputeLossFn(nn.Module):
    def __init__(self, loss_fn: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        output = model(x)
        return self.loss_fn(output, y), output, None


@dataclasses.dataclass
class AdvTrainState(bits.TrainState):
    compute_loss_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], Tuple[
        torch.Tensor, torch.Tensor,
        Optional[torch.Tensor]]] = None  # type: ignore

    @classmethod
    def from_bits(cls, instance: bits.TrainState, **kwargs):
        return cls(model=instance.model,
                   train_loss=instance.train_loss,
                   eval_loss=instance.eval_loss,
                   updater=instance.updater,
                   lr_scheduler=instance.lr_scheduler,
                   model_ema=instance.model_ema,
                   train_cfg=instance.train_cfg,
                   epoch=instance.epoch,
                   step_count=instance.step_count,
                   step_count_global=instance.step_count_global,
                   **kwargs)
