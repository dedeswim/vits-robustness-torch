"""image_net_subset dataset."""

import io
import jax
import numpy as np
import tensorflow_datasets as tfds

_FILES_FNAME = 'test_image_ids.txt'


class RobustbenchImageNet(tfds.image_classification.Imagenet2012):
    def __init__(self, **kwargs):
        original_labels = np.loadtxt(_FILES_FNAME, dtype=np.str)
        self.filenames = set(original_labels)
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
                super_info.features['label'],
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
                if example['label'] in self.filenames:
                    yield key, example

        if validation_labels:  # Validation split
            for key, example in self._generate_examples_validation(
                    archive, validation_labels):
                if example['file_name'] in self.filenames:
                    yield key, example

        # Training split. Main archive contains archives names after a synset noun.
        # Each sub-archive contains pictures associated to that synset.
        for i, (fname, fobj) in enumerate(archive):
            if i > 0:
                break
            label = fname[:-4]  # fname is something like 'n01632458.tar'
            fobj_mem = io.BytesIO(fobj.read())
            for j, (image_fname, image) in enumerate(tfds.download.iter_archive(
                    fobj_mem, tfds.download.ExtractMethod.TAR_STREAM)):
                if j > 0:
                    break
                image = self._fix_image(image_fname, image)
                record = {
                    'file_name': image_fname,
                    'image': image,
                    'label': label,
                }
                yield image_fname, record
