import csv
import dataclasses
import glob
import os
import tempfile
from typing import Callable, Optional, Tuple

import timm
import torch
from tensorflow.io import gfile  # flake8: disable=import-error
from timm import bits
from timm.data import PreprocessCfg
from timm.data.dataset_factory import create_dataset
from torch import nn
from torchvision import datasets

import attacks

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


def load_model_from_gcs(checkpoint_path: str, model_name: str):
    with tempfile.TemporaryDirectory() as dst:
        local_checkpoint_path = os.path.join(dst,
                                             os.path.basename(checkpoint_path))
        gfile.copy(checkpoint_path, local_checkpoint_path)
        model = timm.create_model(model_name,
                                  checkpoint_path=local_checkpoint_path)
    return model


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


class ComputeLossFn(nn.Module):
    def __init__(self, loss_fn: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(
            self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, _: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        output = model(x)
        return self.loss_fn(output, y), output, None


@dataclasses.dataclass
class AdvTrainState(bits.TrainState):
    compute_loss_fn: Callable[
        [nn.Module, torch.Tensor, torch.Tensor, int],
        Tuple[torch.Tensor, torch.Tensor,
              Optional[torch.Tensor]]] = None  # type: ignore
    eps_schedule: attacks.EpsSchedule = None  # type: ignore

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
