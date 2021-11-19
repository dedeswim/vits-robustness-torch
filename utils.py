import csv
import dataclasses
import glob
import os
from typing import Callable, Optional, Tuple

import torch
from tensorflow.io import gfile  # flake8: disable=import-error
from timm import bits
from timm.data import PreprocessCfg
from timm.data.dataset_factory import create_dataset
from torch import nn
from torchvision import datasets

_DATASETS = {'cifar10': datasets.CIFAR10, 'cifar100': datasets.CIFAR100}


def get_outdir(path: str, *paths: str, inc=False) -> str:
    """Adapted to get out dir from GCS"""
    outdir = os.path.join(path, *paths)
    if path.startswith('gs://'):
        os_module = gfile
        exists_fn = lambda x: os_module.exists(x)
    else:
        os_module = os
        exists_fn = os.path.exists
    if not exists_fn(outdir):
        os_module.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while exists_fn(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os_module.makedirs(outdir)
    return outdir


def upload_checkpoints_gcs(checkpoints_dir: str, output_dir: str):
    checkpoints_paths = glob.glob(os.path.join(checkpoints_dir, '*.pth.tar'))
    for checkpoint in checkpoints_paths:
        gcs_checkpoint_path = os.path.join(output_dir,
                                           os.path.basename(checkpoint))
        gfile.copy(checkpoint, gcs_checkpoint_path)


class GCSSummaryCsv(bits.monitor.SummaryCsv):
    """SummaryCSV version to work with GCS"""
    def __init__(self, output_dir, filename='summary.csv'):
        super().__init__(output_dir, filename)

    def update(self, row_dict):
        with gfile.GFile(self.filename, mode='a') as cf:
            dw = csv.DictWriter(cf, fieldnames=row_dict.keys())
            if self.needs_header:  # first iteration (epoch == 1 can't be used)
                dw.writeheader()
                self.needs_header = False
            dw.writerow(row_dict)


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


@dataclasses.dataclass
class MyPreprocessCfg(PreprocessCfg):
    normalize: bool = True


class ScaleIn0To1:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1 / 2
