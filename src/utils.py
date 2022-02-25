import csv
import dataclasses
import glob
import os
import tempfile
from typing import Callable, Optional, Tuple

import tensorflow as tf
import timm
import torch
from timm import bits
from timm.data import PreprocessCfg
from torch import nn

import src.attacks as attacks


def get_outdir(path: str, *paths: str, inc=False) -> str:
    """Adapted to get out dir from GCS"""
    outdir = os.path.join(path, *paths)
    if path.startswith('gs://'):
        os_module = tf.io.gfile
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


def load_model_from_gcs(checkpoint_path: str, model_name: str, **kwargs):
    with tempfile.TemporaryDirectory() as dst:
        local_checkpoint_path = os.path.join(dst, os.path.basename(checkpoint_path))
        tf.io.gfile.copy(checkpoint_path, local_checkpoint_path)
        model = timm.create_model(model_name, checkpoint_path=local_checkpoint_path, **kwargs)
    return model


def load_state_dict_from_gcs(model: nn.Module, checkpoint_path: str):
    with tempfile.TemporaryDirectory() as dst:
        local_checkpoint_path = os.path.join(dst, os.path.basename(checkpoint_path))
        tf.io.gfile.copy(checkpoint_path, local_checkpoint_path)
        model.load_state_dict(torch.load(local_checkpoint_path)["model"])
    return model


def upload_checkpoints_gcs(checkpoints_dir: str, output_dir: str):
    checkpoints_paths = glob.glob(os.path.join(checkpoints_dir, '*.pth.tar'))
    for checkpoint in checkpoints_paths:
        gcs_checkpoint_path = os.path.join(output_dir, os.path.basename(checkpoint))
        tf.io.gfile.copy(checkpoint, gcs_checkpoint_path)


class GCSSummaryCsv(bits.monitor.SummaryCsv):
    """SummaryCSV version to work with GCS"""

    def __init__(self, output_dir, filename='summary.csv'):
        super().__init__(output_dir, filename)

    def update(self, row_dict):
        with tf.io.gfile.GFile(self.filename, mode='a') as cf:
            dw = csv.DictWriter(cf, fieldnames=row_dict.keys())
            if self.needs_header:  # first iteration (epoch == 1 can't be used)
                dw.writeheader()
                self.needs_header = False
            dw.writerow(row_dict)


class ComputeLossFn(nn.Module):

    def __init__(self, loss_fn: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                _: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        output = model(x)
        return self.loss_fn(output, y), output, None


@dataclasses.dataclass
class AdvTrainState(bits.TrainState):
    # pytype: disable=annotation-type-mismatch
    compute_loss_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor, int],
                              Tuple[torch.Tensor, torch.Tensor,
                                    Optional[torch.Tensor]]] = None  # type: ignore
    eps_schedule: attacks.EpsSchedule = None  # type: ignore

    # pytype: enable=annotation-type-mismatch

    @classmethod
    def from_bits(cls, instance: bits.TrainState, **kwargs):
        return cls(
            model=instance.model,  # type: ignore
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
