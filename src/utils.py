from collections import OrderedDict
import csv
import dataclasses
import glob
import os
import tempfile
from typing import Callable, Optional, Tuple, Union

import tensorflow as tf
import timm
import torch
from timm import bits
from timm.data import PreprocessCfg
from timm.data.fetcher import Fetcher
from timm.data.prefetcher_cuda import PrefetcherCuda
from torch import nn

import src.attacks as attacks


def get_outdir(path: str, *paths: str, inc=False) -> str:
    """Adapted to get out dir from GCS"""
    outdir = os.path.join(path, *paths)
    if path.startswith('gs://'):
        check_bucket_zone(path, "robust-vits")
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


def check_bucket_zone(data_dir, prefix):
    if "ZONE" not in os.environ:
        raise ValueError(
            "The zone is not set for this machine, set the ZONE env variable to the zone of the machine")
    zone = os.environ['ZONE']
    if zone == "US":
        assert data_dir.startswith(f"gs://{prefix}-us/"), f"The given dir {data_dir} is in the wrong zone"
    elif zone == "EU":
        assert data_dir.startswith(f"gs://{prefix}/"), f"The given dir {data_dir} is in the wrong zone"


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


class ImageNormalizer(nn.Module):
    """From
    https://github.com/RobustBench/robustbench/blob/master/robustbench/model_zoo/architectures/utils_architectures.py#L8"""

    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input - self.mean) / self.std


def normalize_model(model: nn.Module, mean: Tuple[float, float, float], std: Tuple[float, float,
                                                                                   float]) -> nn.Module:
    """From
    https://github.com/RobustBench/robustbench/blob/master/robustbench/model_zoo/architectures/utils_architectures.py#L20"""
    layers = OrderedDict([('normalize', ImageNormalizer(mean, std)), ('model', model)])
    return nn.Sequential(layers)


class CombinedLoaders:

    def __init__(self, loader_1: Union[Fetcher, PrefetcherCuda], loader_2: Union[Fetcher, PrefetcherCuda]):
        self.loader_1 = loader_1
        self.loader_2 = loader_2
        assert loader_1.mixup_enabled == loader_2.mixup_enabled
        self._mixup_enabled = loader_1.mixup_enabled

    def __iter__(self):
        return self._iterator()

    def __len__(self):
        return min(len(self.loader_1), len(self.loader_2))

    def _iterator(self):
        for (img1, label1), (img2, label2) in zip(self.loader_1, self.loader_2):
            images = torch.cat([img1, img2])
            labels = torch.cat([label1, label2])
            indices = torch.randperm(len(images))
            yield images[indices], labels[indices]

    @property
    def sampler(self):
        return self.loader_1.sampler

    @property
    def sampler2(self):
        return self.loader_2.sampler

    @property
    def mixup_enabled(self):
        return self._mixup_enabled

    @mixup_enabled.setter
    def mixup_enabled(self):
        self.loader_1.mixup_enabled = False
        self.loader_2.mixup_enabled = False
        assert self.loader_1.mixup_enabled == self.loader_2.mixup_enabled
        self._mixup_enabled = False
