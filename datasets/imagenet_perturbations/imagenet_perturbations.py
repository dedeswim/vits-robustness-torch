"""imagenet_perturbations dataset."""

from collections import OrderedDict
import dataclasses
import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import timm
import torch
import torchvision.transforms.functional as F
from timm.bits import initialize_device
from timm.data import create_dataset, create_loader_v2, resolve_data_config, PreprocessCfg, constants
from torch import nn
from torchvision import transforms

import attacks
from adv_resnet import resnet50, EightBN

_DESCRIPTION = """
Adversarial perturbations from the ImageNet dataset.
Only validation examples are computed, so use only for validating.
"""

_CITATION = """
"""

MODELS_TO_NORMALIZE = {"resnet50_nonrobust", "resnet50", "resnet50_fgsm", "xcit_small_12_p16_224_nonrobust"}


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


@dataclasses.dataclass
class ImagenetPerturbationsConfig(tfds.core.BuilderConfig):
    model: str = ""
    checkpoint_path: str = ""
    eps: float = 4 / 255
    steps: int = 100
    dataset_name: str = "tfds/robustbench_image_net"
    norm: str = "linf"
    boundaries: Tuple[float, float] = (0., 1.)
    mean: Optional[Tuple[float, float, float]] = None
    std: Optional[Tuple[float, float, float]] = None
    crop_pct: Optional[float] = None
    pretrained: bool = False


class ImagenetPerturbations(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for imagenet_perturbations dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Same manual download instructions as ImageNet
    """
    BUILDER_CONFIGS = [
        ImagenetPerturbationsConfig(name="resnet50_fgsm",
                                    model="resnet50",
                                    checkpoint_path="gs://robust-vits/external-checkpoints/advres50_gelu.pth",
                                    mean=(0.5, 0.5, 0.5),
                                    std=(0.5, 0.5, 0.5),
                                    steps=1),
        ImagenetPerturbationsConfig(name="xcit_small_12_p16_224_fgsm",
                                    model="xcit_small_12_p16_224",
                                    checkpoint_path="gs://robust-vits/xcit/best.pth.tar",
                                    steps=1),
        ImagenetPerturbationsConfig(name="resnet50",
                                    model="adv_resnet50",
                                    checkpoint_path="gs://robust-vits/external-checkpoints/advres50_gelu.pth",
                                    mean=(0.5, 0.5, 0.5),
                                    std=(0.5, 0.5, 0.5)),
        ImagenetPerturbationsConfig(name="resnet50_nonrobust",
                                    model="resnet50",
                                    pretrained=True,
                                    mean=constants.IMAGENET_DEFAULT_MEAN,
                                    std=constants.IMAGENET_DEFAULT_STD),
        ImagenetPerturbationsConfig(name="xcit_small_12_p16_224",
                                    model="xcit_small_12_p16_224",
                                    checkpoint_path="gs://robust-vits/xcit/best.pth.tar"),
        ImagenetPerturbationsConfig(name="xcit_small_12_p16_224_nonrobust",
                                    model="xcit_small_12_p16_224",
                                    pretrained=True,
                                    mean=constants.IMAGENET_DEFAULT_MEAN,
                                    std=constants.IMAGENET_DEFAULT_STD)
    ]

    BATCH_SIZE = 128

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(None, None, 3)),
                'label': tfds.features.ClassLabel(num_classes=1000),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'label'),  # Set to `None` to disable
            homepage='',
            citation=_CITATION,
        )

    def _split_generators(self, _: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dev_env = initialize_device()

        if self.builder_config.name in {"adv_resnet50", "adv_resnet50_fgsm"}:
            model = load_state_dict_from_gcs(resnet50(norm_layer=EightBN),
                                             self.builder_config.checkpoint_path)
        elif self.builder_config.checkpoint_path:
            model = load_model_from_gcs(self.builder_config.checkpoint_path, self.builder_config.model)
        elif self.builder_config.pretrained:
            model = timm.create_model(self.builder_config.model, pretrained=True)
        else:
            raise ValueError(f"For {self.builder_config.name}, either the checkpoint"
                             "should be specified, or pretrained should be `True`")

        if self.builder_config.name in MODELS_TO_NORMALIZE:
            model = normalize_model(model, self.builder_config.mean, self.builder_config.std)

        model.to(dev_env.device)
        model.eval()

        root = self._original_state["data_dir"]
        original_dataset = create_dataset(self.builder_config.dataset_name, root=root, is_training=False)

        data_config = {
            'mean': self.builder_config.mean,
            'std': self.builder_config.std,
            'crop_pct': self.builder_config.crop_pct
        }

        data_config = resolve_data_config(data_config, model=model)
        pp_cfg = PreprocessCfg(  # type: ignore
            input_size=data_config['input_size'],
            interpolation=data_config['interpolation'],
            crop_pct=data_config['crop_pct'],
            mean=data_config['mean'],
            std=data_config['std'])
        loader = create_loader_v2(original_dataset,
                                  batch_size=self.BATCH_SIZE,
                                  is_training=False,
                                  pp_cfg=pp_cfg,
                                  num_workers=2)
        loader.dataset.transform.transforms[-1] = transforms.ToTensor()

        eps = self.builder_config.eps
        steps = self.builder_config.steps
        step_size = 1.5 * eps / steps
        attack = attacks.make_attack("pgd",
                                     eps,
                                     step_size,
                                     steps,
                                     self.builder_config.norm,
                                     self.builder_config.boundaries,
                                     criterion=nn.NLLLoss(reduction="sum"))

        return {
            'validation': self._generate_examples(loader, model, attack),
        }

    def _generate_examples(self, original_loader, model: nn.Module, attack: attacks.AttackFn):
        """Yields examples."""
        for batch_idx, (x_batch, y_batch) in enumerate(original_loader):
            x_adv_batch = attack(model, x_batch, y_batch)
            # Get perturbation and normalize
            pert_batch = ((x_batch - x_adv_batch) + self.builder_config.eps) / (2 * self.builder_config.eps)

            for sample_idx, (x, y) in enumerate(zip(pert_batch, y_batch)):
                key = batch_idx * self.BATCH_SIZE + sample_idx
                yield key, {
                    'image': np.asarray(F.to_pil_image(x.cpu())),
                    'label': y.cpu().item(),
                }
