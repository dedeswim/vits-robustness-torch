"""imagenet_advex dataset."""

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
from timm.data import create_dataset, create_loader_v2, resolve_data_config, PreprocessCfg
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

MODELS_TO_NORMALIZE = {"adv_resnet50", "deit_small_patch16_224"}


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


@dataclasses.dataclass
class ImagenetAdvexConfig(tfds.core.BuilderConfig):
    model: str = ""
    checkpoint_path: str = ""
    eps: float = 4 / 255
    dataset_name: str = "tfds/robustbench_image_net"
    norm: str = "linf"
    boundaries: Tuple[float, float] = (0., 1.)
    mean: Optional[Tuple[float, float, float]] = None
    std: Optional[Tuple[float, float, float]] = None
    crop_pct: Optional[float] = None
    steps: int = 1


class ImagenetAdvex(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for imagenet_advex dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Same manual download instructions as ImageNet
    """
    BUILDER_CONFIGS = [
        ImagenetAdvexConfig(name="resnet50_fgsm",
                            model="resnet50",
                            checkpoint_path="gs://robust-vits/external-checkpoints/advres50_gelu.pth",
                            boundaries=(-1, 1),
                            mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5),
                            eps=8 / 255),
        ImagenetAdvexConfig(name="xcit_small_12_p16_224_fgsm",
                            model="xcit_small_12_p16_224",
                            checkpoint_path="gs://robust-vits/xcit/best.pth.tar"),
        ImagenetAdvexConfig(name="resnet50_pgd10",
                            model="resnet50",
                            checkpoint_path="gs://robust-vits/external-checkpoints/advres50_gelu.pth",
                            boundaries=(-1, 1),
                            mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5),
                            eps=8 / 255,
                            steps=10),
        ImagenetAdvexConfig(name="xcit_small_12_p16_224_pgd10",
                            model="xcit_small_12_p16_224",
                            checkpoint_path="gs://robust-vits/xcit/best.pth.tar",
                            steps=10)
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
        # TODO: instantiate dataset and loader here for the given model
        # TODO: create model here and pass it to _generate_examples
        dev_env = initialize_device()

        if self.builder_config.model == "resnet50":
            model = load_state_dict_from_gcs(resnet50(norm_layer=EightBN),
                                             self.builder_config.checkpoint_path)
        else:
            model = load_model_from_gcs(self.builder_config.checkpoint_path, self.builder_config.model)
        model.to(dev_env.device)

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
        if self.builder_config.model not in MODELS_TO_NORMALIZE:
            # Do not normalize if model is not from the other paper
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

            for sample_idx, (x, y) in enumerate(zip(x_adv_batch, y_batch)):
                key = batch_idx * self.BATCH_SIZE + sample_idx
                yield key, {
                    'image': np.asarray(F.to_pil_image(x.cpu())),
                    'label': y.cpu().item(),
                }
