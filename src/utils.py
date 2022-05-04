import csv
import dataclasses
import glob
import os
import tempfile
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, Union

import tensorflow as tf
import timm
import torch
import torch.nn.functional as F
from timm import bits
from timm.data import PreprocessCfg
from timm.data.fetcher import Fetcher
from timm.data.prefetcher_cuda import PrefetcherCuda
from timm.models import xcit
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


def write_wandb_info(notes: str, output_dir: str, wandb_run):
    assert output_dir is not None
    # Log run notes and *true* output dir to wandb
    if output_dir.startswith("gs://"):
        exp_dir = output_dir.split("gs://")[-1]
        bucket_url = f"https://console.cloud.google.com/storage/{exp_dir}"
        notes += f"Bucket: {exp_dir}\n"
        wandb_run.config.update({"output": bucket_url}, allow_val_change=True)
    else:
        wandb_run.config.update({"output": output_dir}, allow_val_change=True)
    wandb_run.notes = notes
    wandb_run_field = f"wandb_run: {wandb_run.url}\n"  # type: ignore
    # Log wandb run url to args file
    with tf.io.gfile.GFile(os.path.join(output_dir, 'args.yaml'), 'a') as f:
        f.write(wandb_run_field)


def interpolate_position_embeddings(model: nn.Module, checkpoint_model: Dict[str, Any]):
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens)**0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches**0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    return new_pos_embed


def adapt_model_patches(model: xcit.XCiT, new_patch_size: int):
    to_divide = model.patch_embed.patch_size / new_patch_size
    assert int(to_divide) == to_divide, "The new patch size should divide the original patch size"
    to_divide = int(to_divide)
    assert to_divide % 2 == 0, "The ratio between the original patch size and the new patch size should be divisible by 2"
    for conv_index in range(0, to_divide, 2):
        model.patch_embed.proj[conv_index][0].stride = (1, 1)
    model.patch_embed.patch_size = new_patch_size
    return model
