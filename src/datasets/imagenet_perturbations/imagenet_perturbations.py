"""imagenet_perturbations dataset."""

import dataclasses
import os
from statistics import mode
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import timm
import torchvision.transforms.functional as F
from timm.bits import initialize_device
from timm.data import create_dataset, create_loader_v2, resolve_data_config, PreprocessCfg
from torchvision import transforms

# TODO(imagenet_perturbations): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Adversarial perturbations from the ImageNet dataset.
Only validation examples are computed, so use only for validating.
"""

# TODO(imagenet_perturbations): BibTeX citation
_CITATION = """
"""


def load_model_from_gcs(checkpoint_path: str, model_name: str, **kwargs):
    with tempfile.TemporaryDirectory() as dst:
        local_checkpoint_path = os.path.join(dst, os.path.basename(checkpoint_path))
        tf.io.gfile.copy(checkpoint_path, local_checkpoint_path)
        model = timm.create_model(model_name, checkpoint_path=local_checkpoint_path, **kwargs)
    return model


@dataclasses.dataclass
class ImagenetPerturbationsConfig(tfds.core.BuilderConfig):
    checkpoint_path: str = ""


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
        ImagenetPerturbationsConfig(name="resnet50", checkpoint_path="gs://robust-vits/xcit-3/best.pth.tar")
    ]

    BATCH_SIZE = 128
    ORIGINAL_DATASET_NAME = "tfds/robustbench_image_net"
    ATTACK_STEPS = 100

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

        model = load_model_from_gcs(self.builder_config.checkpoint_path, self.builder_config.name)
        model.to(dev_env.device)

        root = self._original_state["data_dir"]
        original_dataset = create_dataset(self.ORIGINAL_DATASET_NAME, root=root, is_training=False)

        data_config = resolve_data_config({}, model=model)
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

        return {
            'validation': self._generate_examples(model, loader),
        }

    def _generate_examples(self, model, original_loader):
        """Yields examples."""
        for batch_idx, (x_batch, y_batch) in enumerate(original_loader):
            # TODO: create examples here
            pert_batch = x_batch

            for sample_idx, (x, y) in enumerate(zip(pert_batch, y_batch)):
                key = batch_idx * self.BATCH_SIZE + sample_idx
                yield key, {
                    'image': np.asarray(F.to_pil_image(x.cpu())),
                    'label': y.cpu().item(),
                }
