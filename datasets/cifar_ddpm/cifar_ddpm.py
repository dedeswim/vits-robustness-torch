"""cifar_ddpm dataset."""
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
Synthetic data as a proxy distribution to CIFAR-10
"""

_CITATION = """
@article{sehwag2021robust,
  title={Robust Learning Meets Generative Models: Can Proxy Distributions Improve Adversarial Robustness?},
  author={Sehwag, Vikash and Mahloujifar, Saeed and Handina, Tinashe and Dai, Sihui and Xiang, Chong and Chiang, Mung and Mittal, Prateek},
  journal={arXiv preprint arXiv:2104.09425},
  year={2021}
}
"""


class CifarDdpm(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cifar_ddpm dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    LEN = 15402688
    DIRECTORY = Path("cifar10_ddpm_serialized")
    IMGS_FILENAME = "cifar_ddpm_improvedddpm_sorted_images.bin"
    LABELS_FILENAME = "cifar_ddpm_improvedddpm_sorted_labels.npy"
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Download the data and place them in "<manual_dir>/cifar10_ddpm_serialized"
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(32, 32, 3)),
                'label': tfds.features.ClassLabel(num_classes=10),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'label'),  # Set to `None` to disable
            homepage='',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        imgs_path = dl_manager.manual_dir / self.DIRECTORY / self.IMGS_FILENAME
        labels_path = dl_manager.manual_dir / self.DIRECTORY / self.LABELS_FILENAME
        with tf.io.gfile.GFile(labels_path, "rb") as df:
            labels = np.load(df)

        return {
            'train': self._generate_examples(imgs_path, labels),
        }

    def _sample_image(self, df, idx):
        df.seek(idx * 3072)
        image = np.array(np.frombuffer(df.read(3072), dtype="uint8").reshape(32, 32, 3))
        return image

    def _generate_examples(self, imgs_path, labels):
        """Yields examples."""
        with tf.io.gfile.GFile(imgs_path, "rb") as df:
            for idx in range(self.LEN):
                img = self._sample_image(df, idx)
                yield idx, {
                    'image': img,
                    'label': labels[idx],
                }
