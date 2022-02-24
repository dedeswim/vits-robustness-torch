"""image_net_subset dataset."""

import io
import os
import jax
import numpy as np
import tensorflow_datasets as tfds

_LABELS_FNAME = 'imagenet2012_labels.txt'


class ImageNetSubset(tfds.image_classification.Imagenet2012):
    def __init__(self, num_classes: int = 100, seed=0, **kwargs):
        rng = jax.random.PRNGKey(seed)
        self.num_classes = num_classes
        original_labels = np.loadtxt(_LABELS_FNAME, dtype=np.str)
        labels_idx = jax.random.choice(rng,
                                       1_000, (num_classes, ),
                                       replace=False)
        self.labels = set(original_labels[labels_idx])
        assert len(self.labels) == num_classes
        super().__init__(**kwargs)

    def _info(self):
        super_info = super()._info()
        return tfds.core.DatasetInfo(
            builder=self,
            description=super_info.description,
            features=tfds.features.FeaturesDict({
                'image':
                super_info.features['image'],
                'label':
                tfds.features.ClassLabel(names=self.labels),
                'file_name':
                super_info.features['file_name'],
            }),
            supervised_keys=super_info.supervised_keys,
            homepage=super_info.homepage,
            citation=super_info.citation,
        )

    def _generate_examples(self,
                           archive,
                           validation_labels=None,
                           labels_exist=True):
        """Yields examples."""
        if not labels_exist:  # Test split
            for key, example in self._generate_examples_test(archive):
                if example['label'] in self.labels:
                    yield key, example

        if validation_labels:  # Validation split
            for key, example in self._generate_examples_validation(
                    archive, validation_labels):
                if example['label'] in self.labels:
                    yield key, example

        # Training split. Main archive contains archives names after a synset noun.
        # Each sub-archive contains pictures associated to that synset.
        for fname, fobj in archive:
            label = fname[:-4]  # fname is something like 'n01632458.tar'
            if label not in self.labels:
                continue
            fobj_mem = io.BytesIO(fobj.read())
            for image_fname, image in tfds.download.iter_archive(
                    fobj_mem, tfds.download.ExtractMethod.TAR_STREAM):
                image = self._fix_image(image_fname, image)
                record = {
                    'file_name': image_fname,
                    'image': image,
                    'label': label,
                }
                yield image_fname, record
