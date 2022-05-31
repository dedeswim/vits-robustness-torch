"""Dataset wrapper to perform AugMix or other clean/augmentation mixes.

Parts of this file are adapted from PyTorch Image Models by Ross Wightman

The original ones can be found at https://github.com/rwightman/pytorch-image-models/

The original license can be found at this link:
https://github.com/rwightman/pytorch-image-models/blob/master/LICENSE
"""

import torch.utils.data as data
from timm.data.dataset import AugMixDataset


class IterableAugMixDataset(data.IterableDataset, AugMixDataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __iter__(self):
        for x, y in self.dataset:  # all splits share the same dataset base transform
            x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
            # run the full augmentation on the remaining splits
            for _ in range(self.num_splits - 1):
                x_list.append(self._normalize(self.augmentation(x)))

            for x_i in x_list:
                yield x_i, y

    def __len__(self):
        return len(self.dataset) * self.num_splits
